"""
pointcloud.py — RGB-D to colored point cloud conversion.
Converts registered depth + color frames into compact 3D point arrays
suitable for real-time WebSocket streaming.

Supports Kinect v1 (640x480) and v2 (1920x1080) with per-camera intrinsics.
"""
import numpy as np
from typing import Optional, Tuple

from platform_backend import DeviceInfo


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    info: DeviceInfo,
    stride: int = 4,
    depth_min_mm: Optional[float] = None,
    depth_max_mm: Optional[float] = None,
    max_points: int = 50000,
    color_mode: str = "depth",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an RGB-D frame pair into a colored point cloud.

    Args:
        rgb: BGR image (H, W, 3) uint8 (used only when color_mode="rgb")
        depth: Registered depth map (H, W) float32, millimeters
        info: Camera intrinsics (DeviceInfo)
        stride: Downsample factor (e.g. 4 = every 4th pixel in each axis)
        depth_min_mm: Minimum valid depth (default from DeviceInfo)
        depth_max_mm: Maximum valid depth (default from DeviceInfo)
        max_points: Hard cap on output points (random subsample if exceeded)
        color_mode: "depth" (cool-to-warm by distance) or "rgb" (camera color)

    Returns:
        positions: (N, 3) float32 array of XYZ in meters
        colors: (N, 3) uint8 array of RGB colors
    """
    h, w = depth.shape[:2]

    if depth_min_mm is None:
        depth_min_mm = info.depth_min_mm
    if depth_max_mm is None:
        depth_max_mm = info.depth_max_mm

    # Downsample by stride
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy_flat = yy.ravel()
    xx_flat = xx.ravel()

    # Sample depth
    d = depth[yy_flat, xx_flat]

    # Valid depth mask
    valid = (d >= depth_min_mm) & (d <= depth_max_mm) & (d > 0)
    yy_v = yy_flat[valid]
    xx_v = xx_flat[valid]
    d_v = d[valid]

    # Unproject to 3D using intrinsics
    fx = info.fx
    fy = info.fy
    cx = info.cx
    cy = info.cy

    # Scale intrinsics if depth map differs from native resolution
    scale_x = w / info.width
    scale_y = h / info.height
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    z_m = d_v / 1000.0
    x_m = (xx_v.astype(np.float32) - cx) * z_m / fx
    # Negate Y: image rows increase downward, but Three.js/SDK use Y-up
    y_m = -((yy_v.astype(np.float32) - cy) * z_m / fy)

    positions = np.stack([x_m, y_m, z_m], axis=1).astype(np.float32)

    # Generate colors
    if color_mode == "rgb" and rgb is not None:
        rgb_h, rgb_w = rgb.shape[:2]
        if rgb_h != h or rgb_w != w:
            cy_rgb = (yy_v.astype(np.float32) * rgb_h / h).astype(np.int32)
            cx_rgb = (xx_v.astype(np.float32) * rgb_w / w).astype(np.int32)
            cy_rgb = np.clip(cy_rgb, 0, rgb_h - 1)
            cx_rgb = np.clip(cx_rgb, 0, rgb_w - 1)
        else:
            cy_rgb = yy_v
            cx_rgb = xx_v
        colors_bgr = rgb[cy_rgb, cx_rgb]
        colors = colors_bgr[:, ::-1].copy()
    else:
        # Depth-based coloring: near=cyan, mid=green, far=warm orange
        t = (d_v - depth_min_mm) / max(depth_max_mm - depth_min_mm, 1.0)
        t = np.clip(t, 0.0, 1.0)
        r = np.clip((t * 2.0) * 200 + 40, 40, 240).astype(np.uint8)
        g = np.clip((1.0 - abs(t - 0.5) * 2.0) * 220 + 30, 30, 250).astype(np.uint8)
        b = np.clip(((1.0 - t) * 2.0) * 200 + 40, 40, 240).astype(np.uint8)
        colors = np.stack([r, g, b], axis=1)

    # Subsample if over max_points
    n = positions.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        positions = positions[idx]
        colors = colors[idx]

    return positions, colors


def transform_pointcloud(
    positions: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 transform matrix to point cloud positions.

    Args:
        positions: (N, 3) float32
        transform: (4, 4) float64/float32 camera-to-world matrix

    Returns:
        (N, 3) float32 transformed positions
    """
    n = positions.shape[0]
    if n == 0:
        return positions
    # Homogeneous coordinates
    ones = np.ones((n, 1), dtype=np.float32)
    pts_h = np.concatenate([positions, ones], axis=1)  # (N, 4)
    pts_world = (transform @ pts_h.T).T[:, :3]  # (N, 3)
    return pts_world.astype(np.float32)


def merge_pointclouds(
    clouds: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple (positions, colors) tuples into one cloud.

    Args:
        clouds: list of (positions, colors) tuples

    Returns:
        (merged_positions, merged_colors)
    """
    if not clouds:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    positions = [c[0] for c in clouds if c[0].shape[0] > 0]
    colors = [c[1] for c in clouds if c[1].shape[0] > 0]
    if not positions:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(positions, axis=0), np.concatenate(colors, axis=0)


def encode_pointcloud_binary(
    positions: np.ndarray,
    colors: np.ndarray,
    skeleton: Optional[dict] = None,
    camera_id: int = -1,
    timestamp: float = 0.0,
) -> bytes:
    """
    Encode a point cloud + skeleton into a compact binary frame for WebSocket.

    Binary format:
        Header (20 bytes):
            magic: uint32 = 0x50434C44 ("PCLD")
            camera_id: int32 (-1 = fused)
            timestamp: float64
            num_points: uint32
        Point data (num_points * 15 bytes):
            x, y, z: float32 (12 bytes)
            r, g, b: uint8 (3 bytes)
        Skeleton JSON (remaining bytes):
            utf-8 encoded JSON string (or empty)

    Args:
        positions: (N, 3) float32
        colors: (N, 3) uint8
        skeleton: dict of joint data (optional)
        camera_id: -1 for fused, 0+ for per-camera
        timestamp: frame timestamp

    Returns:
        bytes ready for WebSocket binary send
    """
    import struct
    import json

    n = positions.shape[0]

    # Header: magic + cam_id + timestamp + num_points
    header = struct.pack("<IidI", 0x50434C44, camera_id, timestamp, n)

    # Interleave position + color data
    # positions: (N, 3) float32, colors: (N, 3) uint8
    pos_bytes = positions.astype(np.float32).tobytes()
    col_bytes = colors.astype(np.uint8).tobytes()

    # Skeleton JSON
    if skeleton:
        skel_bytes = json.dumps(skeleton, separators=(",", ":")).encode("utf-8")
    else:
        skel_bytes = b""

    # Length prefix for skeleton section
    skel_header = struct.pack("<I", len(skel_bytes))

    return header + pos_bytes + col_bytes + skel_header + skel_bytes


def decode_pointcloud_binary(data: bytes) -> dict:
    """
    Decode a binary point cloud frame (inverse of encode_pointcloud_binary).
    Used for testing.

    Returns:
        dict with keys: camera_id, timestamp, positions, colors, skeleton
    """
    import struct
    import json

    magic, camera_id, timestamp, n = struct.unpack_from("<IidI", data, 0)
    assert magic == 0x50434C44, f"Bad magic: {magic:#x}"

    offset = 20  # header size

    # Positions
    pos_size = n * 3 * 4  # float32
    positions = np.frombuffer(data[offset:offset + pos_size], dtype=np.float32).reshape((n, 3))
    offset += pos_size

    # Colors
    col_size = n * 3  # uint8
    colors = np.frombuffer(data[offset:offset + col_size], dtype=np.uint8).reshape((n, 3))
    offset += col_size

    # Skeleton
    skel_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    skeleton = None
    if skel_len > 0:
        skeleton = json.loads(data[offset:offset + skel_len].decode("utf-8"))

    return {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "num_points": n,
        "positions": positions,
        "colors": colors,
        "skeleton": skeleton,
    }
