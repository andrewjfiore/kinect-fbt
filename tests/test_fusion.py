"""Tests for MultiCameraFusion joint math."""
import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))

from camera import CameraFrame, Landmark3D, JOINT_INDICES, NUM_LANDMARKS
from fusion import MultiCameraFusion, FusedJoint


def _make_frame(cam_id=0, joints=None):
    """Build a CameraFrame with synthetic landmark data."""
    landmarks = [None] * NUM_LANDMARKS
    if joints:
        for name, (x, y, z) in joints.items():
            idx = JOINT_INDICES[name]
            landmarks[idx] = Landmark3D(
                x=x, y=y, z=z,
                visibility=0.99, depth_confidence=0.95,
                index=idx,
            )
    return CameraFrame(
        camera_id=cam_id,
        landmarks=landmarks,
        timestamp=time.monotonic(),
    )


class TestFusionSingleCamera:
    def test_single_camera_identity_calibration(self):
        """Lateral hip offset (x) is preserved after auto-origin calibration."""
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        # Hips at y=1.0 exactly so origin.y = 1.0-1.0 = 0 (no y shift)
        frame = _make_frame(0, {"LEFT_HIP": (0.12, 1.0, 2.0), "RIGHT_HIP": (-0.12, 1.0, 2.0)})
        joints = fusion.update([frame])
        assert "LEFT_HIP" in joints
        lhip = joints["LEFT_HIP"]
        # x should stay near 0.12 (only z offset, not x, assuming symmetric hips)
        assert abs(lhip.x - 0.12) < 0.05
        # y: 1.0 - origin_y; origin_y = hip_y - 1.0 = 0 → output y ≈ 1.0
        assert abs(lhip.y - 1.0) < 0.1

    def test_joints_tracked_count_increases_with_landmarks(self):
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        frame = _make_frame(0, {
            "LEFT_HIP": (0.0, 0.9, 2.0),
            "RIGHT_HIP": (0.0, 0.9, 2.0),
            "LEFT_ANKLE": (0.2, 0.1, 2.0),
            "RIGHT_ANKLE": (-0.2, 0.1, 2.0),
        })
        fusion.update([frame])
        assert fusion.joints_tracked_count() >= 4

    def test_empty_frame_does_not_crash(self):
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        frame = _make_frame(0)  # no landmarks
        joints = fusion.update([frame])
        assert isinstance(joints, dict)


class TestFusionTwoCamera:
    def test_two_cameras_weighted_average(self):
        cal = {0: np.eye(4), 1: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        # Camera 0 sees hip at x=0.1, camera 1 sees it at x=-0.1 → expected fused ~0.0
        f0 = _make_frame(0, {"LEFT_HIP": (0.1, 1.0, 2.0), "RIGHT_HIP": (0.0, 1.0, 2.0)})
        f1 = _make_frame(1, {"LEFT_HIP": (-0.1, 1.0, 2.0), "RIGHT_HIP": (0.0, 1.0, 2.0)})
        joints = fusion.update([f0, f1])
        lhip = joints["LEFT_HIP"]
        assert abs(lhip.x) < 0.15  # fused closer to 0

    def test_transform_shifts_joints(self):
        """Calibration transform is applied before fusion.

        Camera 0 and camera 1 (offset +1m in X) both report the same world-space
        joint position.  Camera 1 local x=-1 → world x=0 (same as cam 0 local x=0).
        The fused output should agree.
        """
        T = np.eye(4)
        T[0, 3] = 1.0  # camera 1 is +1m on world X
        cal = {0: np.eye(4), 1: T}
        fusion = MultiCameraFusion(cal)
        # cam 0 sees hip at world (0.12, 1.0, 2.0); cam 1 sees same hip at local (-0.88, 1.0, 2.0)
        # → world: (-0.88 + 1.0, 1.0, 2.0) = (0.12, 1.0, 2.0) ✓
        f0 = _make_frame(0, {"LEFT_HIP": (0.12, 1.0, 2.0), "RIGHT_HIP": (-0.12, 1.0, 2.0)})
        f1 = _make_frame(1, {"LEFT_HIP": (-0.88, 1.0, 2.0), "RIGHT_HIP": (-1.12, 1.0, 2.0)})
        joints = fusion.update([f0, f1])
        lhip = joints["LEFT_HIP"]
        # Both cameras agree on same world position → fused x should stay near 0.12
        assert abs(lhip.x - 0.12) < 0.05, f"Expected ~0.12, got {lhip.x}"


class TestFusionTrackers:
    def test_hip_tracker_present_when_hips_visible(self):
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        frame = _make_frame(0, {
            "LEFT_HIP": (0.1, 1.0, 2.0),
            "RIGHT_HIP": (-0.1, 1.0, 2.0),
        })
        fusion.update([frame])
        trackers = fusion.get_trackers()
        tracker_ids = [t.tracker_id for t in trackers]
        assert 1 in tracker_ids  # hip tracker

    def test_foot_trackers_present_when_ankles_visible(self):
        cal = {0: np.eye(4)}
        fusion = MultiCameraFusion(cal)
        frame = _make_frame(0, {
            "LEFT_HIP": (0.1, 1.0, 2.0),
            "RIGHT_HIP": (-0.1, 1.0, 2.0),
            "LEFT_ANKLE": (0.15, 0.05, 2.1),
            "RIGHT_ANKLE": (-0.15, 0.05, 2.1),
        })
        fusion.update([frame])
        trackers = fusion.get_trackers()
        tracker_ids = [t.tracker_id for t in trackers]
        assert 2 in tracker_ids  # left foot
        assert 3 in tracker_ids  # right foot
