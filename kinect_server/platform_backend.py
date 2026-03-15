"""
Platform-abstracted Kinect backend.
- Kinect v2 (Xbox One): pylibfreenect2 (Linux), pykinect2 (Windows)
- Kinect v1 (Xbox 360): freenect (libfreenect Python bindings)

Design reference: SlimeVR server, KinectToVR (K2EX), Driver4VR
"""
import logging
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ──────────────────────────────────────────────────────────────
# Device type constants
# ──────────────────────────────────────────────────────────────
DEVICE_TYPE_V1 = "v1"
DEVICE_TYPE_V2 = "v2"
DEVICE_TYPE_SYNTHETIC = "synthetic"

# ──────────────────────────────────────────────────────────────
# Kinect v2 (Xbox One) intrinsics - 1920x1080 color camera
# ──────────────────────────────────────────────────────────────
V2_FX, V2_FY, V2_CX, V2_CY = 1081.37, 1081.37, 959.5, 539.5
V2_WIDTH, V2_HEIGHT = 1920, 1080
V2_DEPTH_MIN_MM, V2_DEPTH_MAX_MM = 500, 4500

# Legacy aliases for backward compatibility
FX, FY, CX, CY = V2_FX, V2_FY, V2_CX, V2_CY
DEPTH_MIN_MM, DEPTH_MAX_MM = V2_DEPTH_MIN_MM, V2_DEPTH_MAX_MM

# ──────────────────────────────────────────────────────────────
# Kinect v1 (Xbox 360) intrinsics - 640x480 color camera
# ──────────────────────────────────────────────────────────────
V1_FX, V1_FY, V1_CX, V1_CY = 594.21, 591.04, 339.5, 242.7
V1_WIDTH, V1_HEIGHT = 640, 480
V1_DEPTH_MIN_MM, V1_DEPTH_MAX_MM = 500, 4000


@dataclass
class DeviceInfo:
    """Camera intrinsics and metadata for a Kinect device."""
    device_type: str  # "v1", "v2", "synthetic"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_min_mm: float
    depth_max_mm: float


def get_v1_device_info() -> DeviceInfo:
    return DeviceInfo(
        device_type=DEVICE_TYPE_V1,
        width=V1_WIDTH, height=V1_HEIGHT,
        fx=V1_FX, fy=V1_FY, cx=V1_CX, cy=V1_CY,
        depth_min_mm=V1_DEPTH_MIN_MM, depth_max_mm=V1_DEPTH_MAX_MM,
    )


def get_v2_device_info() -> DeviceInfo:
    return DeviceInfo(
        device_type=DEVICE_TYPE_V2,
        width=V2_WIDTH, height=V2_HEIGHT,
        fx=V2_FX, fy=V2_FY, cx=V2_CX, cy=V2_CY,
        depth_min_mm=V2_DEPTH_MIN_MM, depth_max_mm=V2_DEPTH_MAX_MM,
    )


def get_synthetic_device_info() -> DeviceInfo:
    return DeviceInfo(
        device_type=DEVICE_TYPE_SYNTHETIC,
        width=V2_WIDTH, height=V2_HEIGHT,
        fx=V2_FX, fy=V2_FY, cx=V2_CX, cy=V2_CY,
        depth_min_mm=V2_DEPTH_MIN_MM, depth_max_mm=V2_DEPTH_MAX_MM,
    )


class KinectBackend(ABC):
    """Abstract interface for Kinect hardware access (v1 or v2)."""

    @abstractmethod
    def open(self, device_index: int) -> bool: ...

    @abstractmethod
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (rgb_bgr, depth_float32_mm) or None.
        Resolution depends on device type: 1920x1080 for v2, 640x480 for v1.
        """
        ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def num_devices(self) -> int: ...

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Return device intrinsics and metadata."""
        ...


# ──────────────────────────────────────────────────────────────
# Linux backend: pylibfreenect2 (Kinect v2)
# ──────────────────────────────────────────────────────────────
class LinuxFreenect2Backend(KinectBackend):
    def __init__(self):
        self._device = None
        self._pipeline = None
        self._listener = None
        self._registration = None
        self._fn2 = None

    def get_device_info(self) -> DeviceInfo:
        return get_v2_device_info()

    def num_devices(self) -> int:
        try:
            import pylibfreenect2 as fn2
            fn2.setGlobalLogger(fn2.createConsoleLogger(fn2.LoggerLevel.Warning))
            return fn2.Freenect2().enumerateDevices()
        except Exception as e:
            logger.warning(f"pylibfreenect2 enumerate failed: {e}")
            return 0

    def open(self, device_index: int) -> bool:
        try:
            import pylibfreenect2 as fn2
            fn2.setGlobalLogger(fn2.createConsoleLogger(fn2.LoggerLevel.Warning))
            self._fn2 = fn2

            try:
                self._pipeline = fn2.OpenGLPacketPipeline()
                logger.info(f"[Linux] Camera {device_index}: OpenGL pipeline")
            except Exception:
                logger.warning(f"[Linux] Camera {device_index}: OpenGL failed, using CPU pipeline (depth quality lower)")
                self._pipeline = fn2.CpuPacketPipeline()

            freenect = fn2.Freenect2()
            serial = freenect.getDeviceSerialNumber(device_index)
            self._device = freenect.openDevice(serial, self._pipeline)
            self._registration = fn2.Registration(
                self._device.getIrCameraParams(),
                self._device.getColorCameraParams(),
            )
            types = fn2.FrameType.Color | fn2.FrameType.Depth
            self._listener = fn2.SyncMultiFrameListener(types)
            self._device.setColorFrameListener(self._listener)
            self._device.setIrAndDepthFrameListener(self._listener)
            self._device.start()
            logger.info(f"[Linux] Camera {device_index} opened")
            return True
        except Exception as e:
            logger.error(f"[Linux] Failed to open camera {device_index}: {e}")
            return False

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        fn2 = self._fn2
        frames = self._listener.waitForNewFrame(timeout=100)
        if frames is None:
            return None
        try:
            color_frame = frames[fn2.FrameType.Color]
            depth_frame = frames[fn2.FrameType.Depth]

            rgb = color_frame.asarray(dtype=np.uint8)[:, :, :3].copy()

            undistorted = fn2.Frame(512, 424, 4)
            registered  = fn2.Frame(1920, 1080, 4)
            self._registration.apply(color_frame, depth_frame, undistorted, registered)
            depth = registered.asarray(dtype=np.float32)[:, :, 0].copy()
            return rgb, depth
        finally:
            self._listener.release(frames)

    def close(self):
        if self._device:
            try:
                self._device.stop()
                self._device.close()
            except Exception:
                pass
            self._device = None


# ──────────────────────────────────────────────────────────────
# Linux backend: libfreenect (Kinect v1 / Xbox 360)
# ──────────────────────────────────────────────────────────────
class LinuxFreenectV1Backend(KinectBackend):
    """
    Kinect v1 (Xbox 360 / Kinect for Windows v1) backend using libfreenect.
    Install: pip install freenect
    Requires: libfreenect installed system-wide

    Kinect v1 specs:
    - RGB: 640x480 @ 30fps
    - Depth: 640x480 11-bit raw disparity
    - Valid depth range: ~500-4000mm
    """
    def __init__(self):
        self._device_idx = None
        self._freenect = None

    def get_device_info(self) -> DeviceInfo:
        return get_v1_device_info()

    def num_devices(self) -> int:
        try:
            import freenect
            ctx = freenect.init()
            if ctx is None:
                return 0
            count = freenect.num_devices(ctx)
            freenect.shutdown(ctx)
            return count
        except ImportError:
            logger.debug("freenect not installed (pip install freenect)")
            return 0
        except Exception as e:
            logger.warning(f"freenect enumerate failed: {e}")
            return 0

    def open(self, device_index: int) -> bool:
        try:
            import freenect
            self._freenect = freenect
            self._device_idx = device_index

            # Test device availability by trying to get a frame
            # freenect uses sync_get_video/sync_get_depth which handle open internally
            logger.info(f"[V1] Kinect v1 device {device_index} opened (libfreenect)")
            return True
        except ImportError:
            logger.error("[V1] freenect not installed. Run: pip install freenect")
            return False
        except Exception as e:
            logger.error(f"[V1] Failed to open device {device_index}: {e}")
            return False

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._freenect is None:
            return None

        try:
            freenect = self._freenect

            # Get RGB frame (format=RGB, 640x480)
            # sync_get_video returns (array, timestamp)
            rgb_data, _ = freenect.sync_get_video(self._device_idx)
            if rgb_data is None:
                return None

            # Convert RGB to BGR for OpenCV compatibility
            rgb_bgr = rgb_data[:, :, ::-1].copy()

            # Get depth frame (11-bit raw disparity, 640x480)
            # sync_get_depth returns (array, timestamp) with FREENECT_DEPTH_11BIT format
            depth_raw, _ = freenect.sync_get_depth(self._device_idx)
            if depth_raw is None:
                return None

            # Convert 11-bit disparity to mm using Primesense formula
            # Formula: depth_m = 1.0 / (raw * -0.0030711016 + 3.3309495161)
            depth_raw = depth_raw.astype(np.float32)

            # Invalid pixels: value 2047 = max disparity = no return
            valid_mask = (depth_raw < 2047) & (depth_raw > 0)

            # Primesense disparity to depth formula
            denominator = depth_raw * (-0.0030711016) + 3.3309495161

            # Compute depth in mm with safe division
            depth_mm = np.zeros_like(depth_raw)
            safe_mask = valid_mask & (np.abs(denominator) > 0.001)
            depth_mm[safe_mask] = 1000.0 / denominator[safe_mask]

            # Clamp to valid range and zero out invalid pixels
            depth_mm = np.clip(depth_mm, 0, V1_DEPTH_MAX_MM)
            depth_mm[~valid_mask] = 0.0

            return rgb_bgr, depth_mm.astype(np.float32)

        except Exception as e:
            logger.error(f"[V1] Frame capture error: {e}")
            return None

    def close(self):
        if self._freenect is not None:
            try:
                # freenect sync functions handle cleanup internally
                self._freenect.sync_stop()
            except Exception:
                pass
            self._freenect = None
            self._device_idx = None


# ──────────────────────────────────────────────────────────────
# Windows backend: pykinect2 (Kinect for Windows SDK v2)
# Reference: https://github.com/Kinect/PyKinect2
# ──────────────────────────────────────────────────────────────
class WindowsKinect2Backend(KinectBackend):
    """
    Uses pykinect2 which wraps the official Kinect for Windows SDK v2.
    Install: pip install pykinect2
    Requires: Kinect for Windows SDK 2.0 from Microsoft
    Download SDK: https://www.microsoft.com/en-us/download/details.aspx?id=44561
    """
    def __init__(self):
        self._kinect = None
        self._color_w = 1920
        self._color_h = 1080
        self._depth_w = 512
        self._depth_h = 424
        self._coord_mapper = None

    def get_device_info(self) -> DeviceInfo:
        return get_v2_device_info()

    def num_devices(self) -> int:
        # pykinect2 only supports 1 device per machine (SDK limitation)
        try:
            from pykinect2 import PyKinectRuntime, PyKinectV2
            return 1
        except ImportError:
            return 0

    def open(self, device_index: int) -> bool:
        if device_index > 0:
            logger.warning("Windows Kinect for Windows SDK only supports 1 device per machine. "
                           "For multiple Kinects on Windows, use multiple PCs (not officially supported). "
                           "Ignoring device_index > 0.")
            return False
        try:
            from pykinect2 import PyKinectRuntime, PyKinectV2
            self._kinect = PyKinectRuntime.PyKinectRuntime(
                PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
            )
            self._coord_mapper = self._kinect.CoordinateMapper
            logger.info("[Windows] Kinect v2 opened via Kinect for Windows SDK")
            return True
        except ImportError:
            logger.error(
                "[Windows] pykinect2 not installed. Run: pip install pykinect2\n"
                "Also install Kinect for Windows SDK 2.0: "
                "https://www.microsoft.com/en-us/download/details.aspx?id=44561"
            )
            return False
        except Exception as e:
            logger.error(f"[Windows] Failed to open Kinect: {e}\n"
                         "Ensure Kinect for Windows SDK 2.0 is installed and Kinect is connected.")
            return False

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._kinect is None:
            return None
        try:
            from pykinect2 import PyKinectV2
            import ctypes

            if not self._kinect.has_new_color_frame() or not self._kinect.has_new_depth_frame():
                return None

            # Color frame: BGRA 1920x1080
            color_raw = self._kinect.get_last_color_frame()
            rgb = color_raw.reshape((self._color_h, self._color_w, 4))[:, :, :3].copy()

            # Depth frame: uint16 512x424 (mm)
            depth_raw = self._kinect.get_last_depth_frame()
            depth_512 = depth_raw.reshape((self._depth_h, self._depth_w)).astype(np.float32)

            # Map depth to color space using coordinate mapper
            # This is the Windows equivalent of libfreenect2's registration
            depth_color = self._map_depth_to_color(depth_512)

            return rgb, depth_color
        except Exception as e:
            logger.error(f"[Windows] Frame capture error: {e}")
            return None

    def _map_depth_to_color(self, depth_512: np.ndarray) -> np.ndarray:
        """
        Map 512x424 depth to 1920x1080 color space using coordinate mapper.
        Uses vectorized numpy operations for performance.
        Reference: https://github.com/Kinect/PyKinect2/blob/master/pykinect2/PyKinectRuntime.py
        """
        try:
            from pykinect2 import PyKinectV2
            import ctypes

            num_color_pts = self._color_w * self._color_h

            # Use CoordinateMapper to get color-to-depth point mapping
            depth_points = (PyKinectV2.DepthSpacePoint * num_color_pts)()
            self._coord_mapper.MapColorFrameToDepthSpace(
                self._depth_w * self._depth_h,
                depth_512.flatten().astype(ctypes.c_ushort),
                num_color_pts,
                depth_points
            )

            # Vectorized extraction: interpret ctypes array as raw float32 buffer
            # Each DepthSpacePoint is 2 floats (x, y) = 8 bytes total
            # Use np.frombuffer for zero-copy access to the memory
            raw_floats = np.frombuffer(
                (ctypes.c_float * (num_color_pts * 2)).from_buffer(depth_points),
                dtype=np.float32
            )
            # Reshape to (N, 2) and extract x, y columns
            pts_2d = raw_floats.reshape((num_color_pts, 2))
            dp_x = pts_2d[:, 0]
            dp_y = pts_2d[:, 1]

            # Round to nearest integer for pixel lookup
            dx = np.round(dp_x).astype(np.int32)
            dy = np.round(dp_y).astype(np.int32)

            # Create mask for valid depth coordinates
            valid = (dx >= 0) & (dx < self._depth_w) & (dy >= 0) & (dy < self._depth_h)

            # Build depth-aligned-to-color image using vectorized indexing
            depth_color = np.zeros(num_color_pts, dtype=np.float32)
            valid_idx = np.where(valid)[0]
            depth_color[valid_idx] = depth_512[dy[valid_idx], dx[valid_idx]]

            return depth_color.reshape((self._color_h, self._color_w))
        except Exception:
            # Fallback: simple resize (less accurate but functional)
            import cv2
            return cv2.resize(depth_512, (self._color_w, self._color_h),
                              interpolation=cv2.INTER_NEAREST)

    def close(self):
        if self._kinect:
            try:
                self._kinect.close()
            except Exception:
                pass
            self._kinect = None


# ──────────────────────────────────────────────────────────────
# Linux backend: freenect (Kinect v1 / Xbox 360 Kinect)
# USB ID: 045e:02ae, RGB 640x480, Depth 640x480 uint16 mm
# ──────────────────────────────────────────────────────────────
class SyntheticBackend(KinectBackend):
    def __init__(self, device_type: str = DEVICE_TYPE_V2):
        self._device_type = device_type

    def get_device_info(self) -> DeviceInfo:
        if self._device_type == DEVICE_TYPE_V1:
            return get_v1_device_info()
        return get_synthetic_device_info()

    def num_devices(self) -> int:
        return 1

    def open(self, device_index: int) -> bool:
        logger.warning(f"[Synthetic] Using fake camera {device_index} — no real hardware")
        return True

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        info = self.get_device_info()
        rgb   = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        depth = np.full((info.height, info.width), 2000.0, dtype=np.float32)
        return rgb, depth

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────
def create_backend(force_synthetic: bool = False, device_type: str = None) -> KinectBackend:
    """
    Create a backend for a specific device type.

    Args:
        force_synthetic: Use synthetic backend (no hardware)
        device_type: Force specific type ("v1", "v2", or None for auto-detect)
    """
    if force_synthetic:
        return SyntheticBackend(device_type or DEVICE_TYPE_V2)

    if device_type == DEVICE_TYPE_V1:
        if IS_LINUX:
            return LinuxFreenectV1Backend()
        else:
            logger.warning("Kinect v1 only supported on Linux, using synthetic")
            return SyntheticBackend(DEVICE_TYPE_V1)

    if device_type == DEVICE_TYPE_V2:
        if IS_WINDOWS:
            return WindowsKinect2Backend()
        elif IS_LINUX:
            return LinuxFreenect2Backend()
        else:
            logger.warning(f"Unsupported OS: {platform.system()}")
            return SyntheticBackend(DEVICE_TYPE_V2)

    # Auto-detect: try v2 first, then v1
    if IS_WINDOWS:
        b = WindowsKinect2Backend()
        if b.num_devices() > 0:
            return b
        logger.warning("Windows: Kinect SDK not available, using synthetic backend")
        return SyntheticBackend()
    elif IS_LINUX:
        # Try v2 first
        b = LinuxFreenect2Backend()
        if b.num_devices() > 0:
            return b
        # Try v1
        b = LinuxFreenectV1Backend()
        if b.num_devices() > 0:
            return b
        logger.warning("Linux: no Kinect libraries available, using synthetic backend")
        return SyntheticBackend()
    else:
        logger.warning(f"Unsupported OS: {platform.system()}, using synthetic backend")
        return SyntheticBackend()


def create_backend_for_device(device_index: int) -> Tuple[KinectBackend, str]:
    """
    Auto-detect and create backend for a specific device index.
    Tries v2 first, then v1.

    Returns:
        (backend, device_type) tuple
    """
    if IS_WINDOWS:
        b = WindowsKinect2Backend()
        if b.num_devices() > device_index:
            return b, DEVICE_TYPE_V2
        return SyntheticBackend(), DEVICE_TYPE_SYNTHETIC

    elif IS_LINUX:
        # Count v2 devices
        v2_backend = LinuxFreenect2Backend()
        v2_count = v2_backend.num_devices()

        if device_index < v2_count:
            return v2_backend, DEVICE_TYPE_V2

        # Check v1 devices (index adjusted for v2 count)
        v1_backend = LinuxFreenectV1Backend()
        v1_count = v1_backend.num_devices()
        v1_index = device_index - v2_count

        if v1_index < v1_count:
            return v1_backend, DEVICE_TYPE_V1

        return SyntheticBackend(), DEVICE_TYPE_SYNTHETIC

    return SyntheticBackend(), DEVICE_TYPE_SYNTHETIC


def count_devices() -> int:
    """Count available Kinect devices on this platform (v1 + v2)."""
    total = 0
    if IS_WINDOWS:
        total += WindowsKinect2Backend().num_devices()
    elif IS_LINUX:
        total += LinuxFreenect2Backend().num_devices()
        total += LinuxFreenectV1Backend().num_devices()
    return total


def count_devices_by_type() -> Tuple[int, int]:
    """
    Count Kinect devices by type.

    Returns:
        (v2_count, v1_count) tuple
    """
    v2_count = 0
    v1_count = 0

    if IS_WINDOWS:
        v2_count = WindowsKinect2Backend().num_devices()
    elif IS_LINUX:
        v2_count = LinuxFreenect2Backend().num_devices()
        v1_count = LinuxFreenectV1Backend().num_devices()

    return v2_count, v1_count


def enumerate_all_devices() -> list:
    """
    Enumerate all connected Kinect devices with type info.
    Returns list of dicts: {"index": int, "type": "v1"|"v2", "backend": KinectBackend}
    """
    devices = []
    idx = 0

    if IS_LINUX:
        # v1 devices
        v1 = LinuxFreenectV1Backend()
        v1_count = v1.num_devices()
        for i in range(v1_count):
            devices.append({"index": idx, "hw_index": i, "type": "v1"})
            idx += 1

        # v2 devices
        v2 = LinuxFreenect2Backend()
        v2_count = v2.num_devices()
        for i in range(v2_count):
            devices.append({"index": idx, "hw_index": i, "type": "v2"})
            idx += 1

    elif IS_WINDOWS:
        v2 = WindowsKinect2Backend()
        if v2.num_devices() > 0:
            devices.append({"index": idx, "hw_index": 0, "type": "v2"})
            idx += 1

    logger.info(f"Enumerated {len(devices)} Kinect device(s): "
                f"v1={sum(1 for d in devices if d['type'] == 'v1')}, "
                f"v2={sum(1 for d in devices if d['type'] == 'v2')}")
    return devices
