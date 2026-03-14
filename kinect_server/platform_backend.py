"""
Platform-abstracted Kinect backend.
- Linux: pylibfreenect2 (libfreenect2)
- Windows: pykinect2 (official Kinect for Windows SDK v2)

Design reference: SlimeVR server, KinectToVR (K2EX), Driver4VR
"""
import logging
import platform
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# Kinect v2 color camera intrinsics (standard values, both platforms use same sensor)
FX, FY, CX, CY = 1081.37, 1081.37, 959.5, 539.5
DEPTH_MIN_MM, DEPTH_MAX_MM = 500, 4500


class KinectBackend(ABC):
    """Abstract interface for Kinect v2 hardware access."""

    @abstractmethod
    def open(self, device_index: int) -> bool: ...

    @abstractmethod
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (rgb_1920x1080_bgr, depth_1920x1080_float32_mm) or None."""
        ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def num_devices(self) -> int: ...


# ──────────────────────────────────────────────────────────────
# Linux backend: pylibfreenect2
# ──────────────────────────────────────────────────────────────
class LinuxFreenect2Backend(KinectBackend):
    def __init__(self):
        self._device = None
        self._pipeline = None
        self._listener = None
        self._registration = None
        self._fn2 = None

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
        Reference: https://github.com/Kinect/PyKinect2/blob/master/pykinect2/PyKinectRuntime.py
        """
        try:
            from pykinect2 import PyKinectV2
            import ctypes

            # Use CoordinateMapper to get depth-to-color point mapping
            depth_points = (PyKinectV2.DepthSpacePoint * (self._depth_w * self._depth_h))()
            self._coord_mapper.MapColorFrameToDepthSpace(
                self._depth_w * self._depth_h,
                depth_512.flatten().astype(ctypes.c_ushort),
                self._color_w * self._color_h,
                depth_points
            )

            # Build depth-aligned-to-color image
            depth_color = np.zeros((self._color_h, self._color_w), dtype=np.float32)
            for cy in range(self._color_h):
                for cx in range(self._color_w):
                    dp = depth_points[cy * self._color_w + cx]
                    dx = int(dp.x + 0.5)
                    dy = int(dp.y + 0.5)
                    if 0 <= dx < self._depth_w and 0 <= dy < self._depth_h:
                        depth_color[cy, cx] = depth_512[dy, dx]
            return depth_color
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
# Synthetic backend (testing without hardware)
# ──────────────────────────────────────────────────────────────
class SyntheticBackend(KinectBackend):
    def num_devices(self) -> int:
        return 1

    def open(self, device_index: int) -> bool:
        logger.warning(f"[Synthetic] Using fake camera {device_index} — no real hardware")
        return True

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rgb   = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.full((1080, 1920), 2000.0, dtype=np.float32)
        return rgb, depth

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def create_backend(force_synthetic: bool = False) -> KinectBackend:
    if force_synthetic:
        return SyntheticBackend()

    if IS_WINDOWS:
        b = WindowsKinect2Backend()
        if b.num_devices() > 0:
            return b
        logger.warning("Windows: Kinect SDK not available, using synthetic backend")
        return SyntheticBackend()
    elif IS_LINUX:
        b = LinuxFreenect2Backend()
        if b.num_devices() > 0:
            return b
        logger.warning("Linux: pylibfreenect2 not available, using synthetic backend")
        return SyntheticBackend()
    else:
        logger.warning(f"Unsupported OS: {platform.system()}, using synthetic backend")
        return SyntheticBackend()


def count_devices() -> int:
    """Count available Kinect devices on this platform."""
    if IS_WINDOWS:
        return WindowsKinect2Backend().num_devices()
    elif IS_LINUX:
        return LinuxFreenect2Backend().num_devices()
    return 0
