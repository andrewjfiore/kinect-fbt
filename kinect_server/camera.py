"""
KinectCamera: per-device capture + depth-informed landmark extraction.
Supports both Kinect v1 (640x480) and v2 (1920x1080) with per-camera intrinsics.
Uses MediaPipe Pose for 2D landmarks.
"""
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from platform_backend import (
    DeviceInfo, get_v2_device_info, get_synthetic_device_info,
    create_backend_for_device, DEVICE_TYPE_V1, DEVICE_TYPE_V2,
    V2_FX, V2_FY, V2_CX, V2_CY, V2_DEPTH_MIN_MM, V2_DEPTH_MAX_MM,
)

logger = logging.getLogger(__name__)

# Platform backend factory — can be overridden by GUI or tests
PLATFORM_BACKEND_FACTORY = None

# Legacy constants for backward compatibility (v2 defaults)
FX = V2_FX
FY = V2_FY
CX = V2_CX
CY = V2_CY
DEPTH_MIN_MM = V2_DEPTH_MIN_MM
DEPTH_MAX_MM = V2_DEPTH_MAX_MM

# MediaPipe landmark count
NUM_LANDMARKS = 33

JOINT_INDICES = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,     "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}


@dataclass
class Landmark3D:
    x: float
    y: float
    z: float
    visibility: float
    depth_confidence: float
    index: int


@dataclass
class CameraFrame:
    camera_id: int
    landmarks: List[Optional[Landmark3D]]  # 33 entries, None if not visible
    timestamp: float
    rgb_preview: Optional[np.ndarray] = None  # full-res for MJPEG
    device_info: Optional[DeviceInfo] = None  # per-camera intrinsics and metadata


class KinectCamera:
    """
    Manages a single Kinect device (v1 or v2).
    Runs capture + MediaPipe extraction in a background thread,
    exposes latest frame via a queue.
    """

    def __init__(self, device_index: int, fps_target: int = 20, device_type: str = None):
        """
        Args:
            device_index: Camera index
            fps_target: Target frames per second
            device_type: Force device type ("v1", "v2") or None for auto-detect
        """
        self.device_index = device_index
        self.fps_target = fps_target
        self.frame_interval = 1.0 / fps_target
        self._device_type = device_type  # None = auto-detect
        self._device_info: Optional[DeviceInfo] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread = None
        self._mp_pose = None
        self._device = None
        self._pipeline = None
        self._registration = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.monotonic()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device intrinsics (available after start())."""
        return self._device_info or get_v2_device_info()

    def start(self):
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"kinect-{self.device_index}")
        self._thread.start()
        logger.info(f"Camera {self.device_index} started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_frame(self, timeout: float = 0.1) -> Optional[CameraFrame]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        return self._fps

    def _init_kinect(self):
        # Use injected platform backend if available (set by GUI or tests)
        global PLATFORM_BACKEND_FACTORY
        if PLATFORM_BACKEND_FACTORY is not None:
            try:
                self._platform_backend = PLATFORM_BACKEND_FACTORY()
                ok = self._platform_backend.open(self.device_index)
                if ok:
                    self._use_platform_backend = True
                    return True
            except Exception as e:
                logger.warning(f"Platform backend failed: {e}, falling back to pylibfreenect2")

        self._use_platform_backend = False
        try:
            import pylibfreenect2 as fn2

            fn2.setGlobalLogger(fn2.createConsoleLogger(fn2.LoggerLevel.Warning))

            # Try OpenGL pipeline first
            try:
                self._pipeline = fn2.OpenGLPacketPipeline()
                logger.info(f"Camera {self.device_index}: using OpenGL pipeline")
            except Exception:
                logger.warning(f"Camera {self.device_index}: OpenGL pipeline failed, falling back to CPU (depth quality will be lower)")
                self._pipeline = fn2.CpuPacketPipeline()

            self._fn2 = fn2
            freenect = fn2.Freenect2()
            num_devices = freenect.enumerateDevices()
            if self.device_index >= num_devices:
                raise RuntimeError(f"Camera {self.device_index} not found ({num_devices} devices available)")

            serial = freenect.getDeviceSerialNumber(self.device_index)
            self._device = freenect.openDevice(serial, self._pipeline)
            self._registration = fn2.Registration(
                self._device.getIrCameraParams(),
                self._device.getColorCameraParams(),
            )

            types = fn2.FrameType.Color | fn2.FrameType.Depth | fn2.FrameType.Ir
            self._listener = fn2.SyncMultiFrameListener(types)
            self._device.setColorFrameListener(self._listener)
            self._device.setIrAndDepthFrameListener(self._listener)
            self._device.start()
            logger.info(f"Camera {self.device_index}: Kinect opened (serial={serial})")
            return True
        except ImportError:
            logger.warning(f"Camera {self.device_index}: pylibfreenect2 not available, using synthetic data")
            return False
        except Exception as e:
            logger.error(f"Camera {self.device_index}: failed to init Kinect: {e}")
            return False

    def _init_mediapipe(self):
        mp_pose = mp.solutions.pose
        self._mp_pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _run(self):
        self._init_mediapipe()
        has_kinect = self._init_kinect()

        while self._running:
            t_start = time.monotonic()
            try:
                frame = self._capture_frame(has_kinect)
                if frame is not None:
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Drop stale frame — never block
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._frame_queue.put_nowait(frame)

                    self._frame_count += 1
                    now = time.monotonic()
                    if now - self._fps_time >= 1.0:
                        self._fps = self._frame_count / (now - self._fps_time)
                        self._frame_count = 0
                        self._fps_time = now
            except Exception as e:
                logger.error(f"Camera {self.device_index} capture error: {e}")

            elapsed = time.monotonic() - t_start
            if elapsed > 0.050:
                # Frame took too long — skip, don't queue
                logger.debug(f"Camera {self.device_index}: frame skipped ({elapsed*1000:.0f}ms > 50ms)")
                continue

            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        if self._device is not None:
            try:
                self._device.stop()
                self._device.close()
            except Exception:
                pass
        if self._mp_pose is not None:
            self._mp_pose.close()

    def _capture_frame(self, has_kinect: bool) -> Optional[CameraFrame]:
        if hasattr(self, '_use_platform_backend') and self._use_platform_backend:
            return self._capture_platform_frame()
        if has_kinect:
            return self._capture_kinect_frame()
        else:
            return self._capture_synthetic_frame()

    def _capture_platform_frame(self) -> Optional[CameraFrame]:
        result = self._platform_backend.get_frames()
        if result is None:
            return None
        rgb_full, depth_registered = result
        # Get device info from backend
        if self._device_info is None:
            self._device_info = self._platform_backend.get_device_info()
        return self._process_frame(rgb_full, depth_registered)

    def _capture_kinect_frame(self) -> Optional[CameraFrame]:
        fn2 = self._fn2
        frames = self._listener.waitForNewFrame(timeout=100)
        if frames is None:
            return None

        try:
            color_frame = frames[fn2.FrameType.Color]
            depth_frame = frames[fn2.FrameType.Depth]

            # RGB: 1920x1080 BGRX
            rgb_full = color_frame.asarray(dtype=np.uint8)[:, :, :3].copy()

            # Depth: 512x424 float32 (mm)
            depth_raw = depth_frame.asarray(dtype=np.float32).copy()

            # Registered depth aligned to color space
            undistorted = fn2.Frame(512, 424, 4)
            registered = fn2.Frame(1920, 1080, 4)
            self._registration.apply(color_frame, depth_frame, undistorted, registered)
            depth_registered = registered.asarray(dtype=np.float32)[:, :, 0]

        finally:
            self._listener.release(frames)

        # pylibfreenect2 is v2 only
        if self._device_info is None:
            self._device_info = get_v2_device_info()
        return self._process_frame(rgb_full, depth_registered)

    def _capture_synthetic_frame(self) -> Optional[CameraFrame]:
        """Synthetic data for testing without Kinect hardware."""
        # Use device_info resolution or default to v2
        if self._device_info is None:
            self._device_info = get_synthetic_device_info()
        info = self._device_info
        rgb_full = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        depth_registered = np.ones((info.height, info.width), dtype=np.float32) * 2000.0
        return self._process_frame(rgb_full, depth_registered)

    def _process_frame(self, rgb_full: np.ndarray, depth_registered: np.ndarray) -> CameraFrame:
        # Use per-camera intrinsics or defaults
        info = self._device_info or get_v2_device_info()
        fx, fy, cx, cy = info.fx, info.fy, info.cx, info.cy

        # Downsample to 640x480 for MediaPipe (regardless of input resolution)
        rgb_small = cv2.resize(rgb_full, (640, 480))
        rgb_mp = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)

        results = self._mp_pose.process(rgb_mp)

        landmarks_3d: List[Optional[Landmark3D]] = [None] * NUM_LANDMARKS

        if results.pose_landmarks:
            h_full, w_full = rgb_full.shape[:2]

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if lm.visibility < 0.5:
                    continue

                # Map back to full-res pixel coordinates
                px = int(lm.x * w_full)
                py = int(lm.y * h_full)
                px = max(0, min(px, w_full - 1))
                py = max(0, min(py, h_full - 1))

                depth_mm, depth_confidence = self._lookup_depth(depth_registered, px, py, info)

                if depth_confidence > 0:
                    # Use per-camera intrinsics for 3D projection
                    x_m = (px - cx) * depth_mm / (fx * 1000.0)
                    y_m = (py - cy) * depth_mm / (fy * 1000.0)
                    z_m = depth_mm / 1000.0
                else:
                    # Monocular fallback using MediaPipe's Z estimate
                    z_m = abs(lm.z) * 2.0  # rough scale
                    x_m = (px - cx) * z_m / (fx / 1000.0)
                    y_m = (py - cy) * z_m / (fy / 1000.0)
                    depth_confidence = 0.1

                landmarks_3d[idx] = Landmark3D(
                    x=x_m, y=y_m, z=z_m,
                    visibility=lm.visibility,
                    depth_confidence=depth_confidence,
                    index=idx,
                )

            # Draw skeleton overlay on full-res preview
            self._draw_skeleton(rgb_full, landmarks_3d, info)

        return CameraFrame(
            camera_id=self.device_index,
            landmarks=landmarks_3d,
            timestamp=time.monotonic(),
            rgb_preview=rgb_full,
            device_info=info,
        )

    def _lookup_depth(self, depth_map: np.ndarray, px: int, py: int,
                       info: DeviceInfo) -> Tuple[float, float]:
        """Look up depth value at pixel with neighborhood search fallback."""
        h, w = depth_map.shape
        depth_min = info.depth_min_mm
        depth_max = info.depth_max_mm

        d = depth_map[py, px]
        if depth_min <= d <= depth_max:
            return float(d), 1.0

        # 5x5 neighborhood search
        for r in range(1, 3):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        nd = depth_map[ny, nx]
                        if depth_min <= nd <= depth_max:
                            return float(nd), 0.5

        return 0.0, 0.0

    def _draw_skeleton(self, img: np.ndarray, landmarks: List[Optional[Landmark3D]],
                       info: DeviceInfo):
        """Draw skeleton overlay using per-camera intrinsics."""
        fx, fy, cx, cy = info.fx, info.fy, info.cx, info.cy
        for lm in landmarks:
            if lm is None:
                continue
            h, w = img.shape[:2]
            z = lm.z if lm.z > 0 else 1
            px = int(lm.x * fx / z + cx)
            py = int(lm.y * fy / z + cy)
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            if lm.depth_confidence >= 0.9:
                color = (0, 255, 0)
            elif lm.depth_confidence >= 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(img, (px, py), 5, color, -1)


def enumerate_kinect_devices() -> int:
    """Return the total number of connected Kinect devices (v1 + v2)."""
    from platform_backend import count_devices, count_devices_by_type
    try:
        total = count_devices()
        v2, v1 = count_devices_by_type()
        if v1 > 0 or v2 > 0:
            logger.info(f"Found {total} Kinect device(s): {v2} v2, {v1} v1")
        else:
            logger.warning("No Kinect hardware found — defaulting to 1 synthetic camera")
            return 1
        return total
    except Exception as e:
        logger.error(f"Error enumerating Kinect devices: {e}")
        return 1
