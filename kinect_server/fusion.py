"""
MultiCameraFusion: spatial transform + weighted multi-camera fusion + virtual tracker derivation.
"""
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from camera import CameraFrame, Landmark3D, JOINT_INDICES
from filter import OneEuroFilter3D

logger = logging.getLogger(__name__)

STALENESS_TIMEOUT = 0.5  # seconds


@dataclass
class FusedJoint:
    name: str
    x: float
    y: float
    z: float
    confidence: float
    last_seen: float = field(default_factory=time.monotonic)
    is_lost: bool = False


@dataclass
class TrackerData:
    tracker_id: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]  # pitch, yaw, roll (degrees)
    confidence: float


class MultiCameraFusion:
    """
    Fuses joint observations from multiple cameras into world-space joint positions,
    applies 1-euro filters, derives virtual trackers for VRChat FBT.
    """

    def __init__(self, calibration: Dict, user_height: float = 1.7):
        self.calibration = calibration  # {cam_id: 4x4 np.ndarray}
        self.user_height = user_height
        self._joints: Dict[str, FusedJoint] = {
            name: FusedJoint(name=name, x=0, y=0, z=0, confidence=0)
            for name in JOINT_INDICES
        }
        self._filters: Dict[str, OneEuroFilter3D] = {
            name: OneEuroFilter3D(min_cutoff=1.0, beta=0.1)
            for name in JOINT_INDICES
        }
        self._origin: Optional[np.ndarray] = None  # 3D offset applied to all output
        self._height_scale: float = 1.0
        self._calibrated_origin = False
        self._lock = threading.Lock()  # protects _origin, _height_scale, _calibrated_origin

    def update(self, frames: List[CameraFrame]) -> Dict[str, FusedJoint]:
        """Fuse new frames from all cameras, return updated joint dict."""
        now = time.monotonic()

        # Collect per-joint observations across cameras
        observations: Dict[str, List[Tuple[np.ndarray, float]]] = {name: [] for name in JOINT_INDICES}

        for frame in frames:
            cam_id = frame.camera_id
            transform = self.calibration.get(cam_id, np.eye(4))
            for name, idx in JOINT_INDICES.items():
                lm = frame.landmarks[idx]
                if lm is None:
                    continue
                weight = lm.visibility * lm.depth_confidence
                if weight < 0.1:
                    continue
                # Transform camera-local 3D point to world space
                pt_cam = np.array([lm.x, lm.y, lm.z, 1.0])
                pt_world = transform @ pt_cam
                observations[name].append((pt_world[:3], weight))

        # Fuse and filter
        for name, obs in observations.items():
            joint = self._joints[name]
            if len(obs) == 0:
                # Check staleness
                if now - joint.last_seen > STALENESS_TIMEOUT:
                    joint.is_lost = True
                    joint.confidence *= 0.9  # decay
                continue

            if len(obs) == 1:
                pos, w = obs[0]
                confidence = min(w, 1.0)
            else:
                # Weighted average using numpy (avoids sum() starting from int 0)
                positions = np.array([p for p, _ in obs])
                weights = np.array([w for _, w in obs])
                total_w = weights.sum()
                pos = np.average(positions, axis=0, weights=weights)
                confidence = min(total_w / len(obs), 1.0)

            # Apply 1-euro filter
            t = time.monotonic()
            fx, fy, fz = self._filters[name].filter(pos[0], pos[1], pos[2], t)

            joint.x = fx
            joint.y = fy
            joint.z = fz
            joint.confidence = confidence
            joint.last_seen = now
            joint.is_lost = False

        # Auto-calibrate origin on first valid frame
        with self._lock:
            need_calibrate = not self._calibrated_origin
        if need_calibrate:
            self._try_set_origin()

        # Return a copy with origin offset and height scale applied
        # (never modify self._joints in-place — that causes cumulative drift)
        with self._lock:
            origin = self._origin.copy() if self._origin is not None else None
            scale = self._height_scale

        if origin is not None:
            output = {}
            for name, joint in self._joints.items():
                output[name] = FusedJoint(
                    name=joint.name,
                    x=(joint.x - origin[0]) * scale,
                    y=(joint.y - origin[1]) * scale,
                    z=(joint.z - origin[2]) * scale,
                    confidence=joint.confidence,
                    last_seen=joint.last_seen,
                    is_lost=joint.is_lost,
                )
            return output

        return {name: FusedJoint(
            name=j.name, x=j.x, y=j.y, z=j.z,
            confidence=j.confidence, last_seen=j.last_seen, is_lost=j.is_lost,
        ) for name, j in self._joints.items()}

    def recalibrate_origin(self):
        """Reset the world origin (called on /calibrate OSC message)."""
        with self._lock:
            self._calibrated_origin = False
        self._try_set_origin()

    def _try_set_origin(self):
        lhip = self._joints.get("LEFT_HIP")
        rhip = self._joints.get("RIGHT_HIP")
        if lhip and rhip and lhip.confidence > 0.3 and rhip.confidence > 0.3:
            hip_x = (lhip.x + rhip.x) / 2
            hip_y = (lhip.y + rhip.y) / 2
            hip_z = (lhip.z + rhip.z) / 2
            # Target: HIP_CENTER = (0, 1.0, 0)
            new_origin = np.array([hip_x, hip_y - 1.0, hip_z])
            new_scale = self._compute_height_scale()
            with self._lock:
                self._origin = new_origin
                self._height_scale = new_scale
                self._calibrated_origin = True
            logger.info(f"Origin set: {new_origin}, height_scale={new_scale:.3f}")

    def _compute_height_scale(self) -> float:
        """Estimate scale from observed ankle-to-shoulder distance vs expected."""
        try:
            ls = self._joints["LEFT_SHOULDER"]
            rs = self._joints["RIGHT_SHOULDER"]
            la = self._joints["LEFT_ANKLE"]
            ra = self._joints["RIGHT_ANKLE"]
            if all(j.confidence > 0.3 for j in [ls, rs, la, ra]):
                shoulder_y = (ls.y + rs.y) / 2
                ankle_y = (la.y + ra.y) / 2
                observed = abs(shoulder_y - ankle_y)
                # Expected: ~85% of user height from ankle to shoulder
                expected = self.user_height * 0.85
                if observed > 0.1:
                    return expected / observed
        except Exception:
            pass
        return 1.0

    def get_trackers(self) -> List[TrackerData]:
        """Derive the 5 virtual trackers for VRChat FBT.
        Uses the origin-corrected joint positions (same as update() return value).
        """
        trackers = []
        with self._lock:
            origin = self._origin.copy() if self._origin is not None else None
            scale = self._height_scale

        if origin is not None:
            # Build origin-corrected view of joints
            joints = {
                name: FusedJoint(
                    name=j.name,
                    x=(j.x - origin[0]) * scale,
                    y=(j.y - origin[1]) * scale,
                    z=(j.z - origin[2]) * scale,
                    confidence=j.confidence,
                    last_seen=j.last_seen,
                    is_lost=j.is_lost,
                )
                for name, j in self._joints.items()
            }
        else:
            joints = self._joints

        # Tracker 1: HIP_CENTER
        lhip = joints.get("LEFT_HIP")
        rhip = joints.get("RIGHT_HIP")
        if lhip and rhip and not (lhip.is_lost and rhip.is_lost):
            hip_pos = (
                (lhip.x + rhip.x) / 2,
                (lhip.y + rhip.y) / 2,
                (lhip.z + rhip.z) / 2,
            )
            hip_conf = (lhip.confidence + rhip.confidence) / 2
            # Yaw from hip vector on XZ plane
            dx = rhip.x - lhip.x
            dz = rhip.z - lhip.z
            yaw = math.degrees(math.atan2(dx, dz))
            trackers.append(TrackerData(1, hip_pos, (0.0, yaw, 0.0), hip_conf))

        hip_yaw = trackers[0].rotation[1] if trackers else 0.0

        # Tracker 2: LEFT_FOOT
        la = joints.get("LEFT_ANKLE")
        lh = joints.get("LEFT_HEEL")
        if la and not la.is_lost:
            if lh and not lh.is_lost:
                dx = la.x - lh.x
                dz = la.z - lh.z
                yaw = math.degrees(math.atan2(dx, dz))
            else:
                yaw = hip_yaw
            trackers.append(TrackerData(2, (la.x, la.y, la.z), (0.0, yaw, 0.0), la.confidence))

        # Tracker 3: RIGHT_FOOT
        ra = joints.get("RIGHT_ANKLE")
        rh = joints.get("RIGHT_HEEL")
        if ra and not ra.is_lost:
            if rh and not rh.is_lost:
                dx = ra.x - rh.x
                dz = ra.z - rh.z
                yaw = math.degrees(math.atan2(dx, dz))
            else:
                yaw = hip_yaw
            trackers.append(TrackerData(3, (ra.x, ra.y, ra.z), (0.0, yaw, 0.0), ra.confidence))

        # Tracker 4: LEFT_KNEE (optional)
        lk = joints.get("LEFT_KNEE")
        if lk and not lk.is_lost and lk.confidence > 0.3:
            trackers.append(TrackerData(4, (lk.x, lk.y, lk.z), (0.0, hip_yaw, 0.0), lk.confidence))

        # Tracker 5: RIGHT_KNEE (optional)
        rk = joints.get("RIGHT_KNEE")
        if rk and not rk.is_lost and rk.confidence > 0.3:
            trackers.append(TrackerData(5, (rk.x, rk.y, rk.z), (0.0, hip_yaw, 0.0), rk.confidence))

        return trackers

    def joints_tracked_count(self) -> int:
        return sum(1 for j in self._joints.values() if not j.is_lost and j.confidence > 0.2)
