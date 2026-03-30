"""
End-to-end tests for the volumetric 3D streaming pipeline.
Tests: point cloud generation, binary encoding/decoding, skeleton dict building,
point cloud merging, and the full synthetic → pointcloud → encode → decode chain.

No hardware, network, or GPU required.
"""
import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))

from platform_backend import DeviceInfo, get_v1_device_info, get_v2_device_info, get_synthetic_device_info
from pointcloud import (
    rgbd_to_pointcloud,
    transform_pointcloud,
    merge_pointclouds,
    encode_pointcloud_binary,
    decode_pointcloud_binary,
)


# ── Fixtures ──

def _make_rgb_depth(width, height, depth_val_mm=2000.0):
    """Create synthetic RGB + depth frames."""
    rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), depth_val_mm, dtype=np.float32)
    return rgb, depth


def _make_device_info_v2():
    return get_v2_device_info()


def _make_device_info_v1():
    return get_v1_device_info()


# ── Point Cloud Generation ──

class TestRGBDToPointcloud:
    def test_basic_v2(self):
        """V2 frame produces non-empty point cloud."""
        info = _make_device_info_v2()
        rgb, depth = _make_rgb_depth(info.width, info.height, 2000.0)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=8)
        assert pos.shape[0] > 0
        assert pos.shape[1] == 3
        assert col.shape == pos.shape
        assert pos.dtype == np.float32
        assert col.dtype == np.uint8

    def test_basic_v1(self):
        """V1 frame produces non-empty point cloud."""
        info = _make_device_info_v1()
        rgb, depth = _make_rgb_depth(info.width, info.height, 1500.0)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=4)
        assert pos.shape[0] > 0
        assert col.shape[0] == pos.shape[0]

    def test_stride_reduces_points(self):
        """Higher stride = fewer points."""
        info = _make_device_info_v2()
        rgb, depth = _make_rgb_depth(info.width, info.height, 2000.0)
        pos_s2, _ = rgbd_to_pointcloud(rgb, depth, info, stride=2)
        pos_s8, _ = rgbd_to_pointcloud(rgb, depth, info, stride=8)
        assert pos_s2.shape[0] > pos_s8.shape[0]

    def test_max_points_cap(self):
        """max_points limits output size."""
        info = _make_device_info_v2()
        rgb, depth = _make_rgb_depth(info.width, info.height, 2000.0)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=2, max_points=1000)
        assert pos.shape[0] <= 1000

    def test_zero_depth_excluded(self):
        """Zero-depth pixels produce no points."""
        info = _make_device_info_v2()
        rgb = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        depth = np.zeros((info.height, info.width), dtype=np.float32)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=4)
        assert pos.shape[0] == 0

    def test_out_of_range_excluded(self):
        """Depth values outside valid range produce no points."""
        info = _make_device_info_v2()
        rgb = np.zeros((info.height, info.width, 3), dtype=np.uint8)
        depth = np.full((info.height, info.width), 99999.0, dtype=np.float32)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=4)
        assert pos.shape[0] == 0

    def test_depth_to_meters(self):
        """Points are in meters (depth 2000mm → z ≈ 2.0m)."""
        info = _make_device_info_v2()
        rgb, depth = _make_rgb_depth(info.width, info.height, 2000.0)
        pos, _ = rgbd_to_pointcloud(rgb, depth, info, stride=8)
        z_mean = np.mean(pos[:, 2])
        assert abs(z_mean - 2.0) < 0.01

    def test_mismatched_rgb_depth_resolution(self):
        """Handle RGB at different resolution than depth."""
        info = _make_device_info_v2()
        # RGB at full res, depth at half res
        rgb = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        depth = np.full((540, 960), 2000.0, dtype=np.float32)
        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=4)
        assert pos.shape[0] > 0


# ── Transform ──

class TestTransformPointcloud:
    def test_identity(self):
        """Identity transform leaves points unchanged."""
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = transform_pointcloud(pts, np.eye(4))
        np.testing.assert_allclose(result, pts, atol=1e-5)

    def test_translation(self):
        """Translation moves points."""
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        result = transform_pointcloud(pts, T)
        np.testing.assert_allclose(result, [[1, 2, 3]], atol=1e-5)

    def test_empty(self):
        """Empty input returns empty output."""
        pts = np.empty((0, 3), dtype=np.float32)
        result = transform_pointcloud(pts, np.eye(4))
        assert result.shape == (0, 3)


# ── Merge ──

class TestMergePointclouds:
    def test_merge_two(self):
        """Merging two clouds concatenates them."""
        p1 = np.array([[1, 2, 3]], dtype=np.float32)
        c1 = np.array([[255, 0, 0]], dtype=np.uint8)
        p2 = np.array([[4, 5, 6]], dtype=np.float32)
        c2 = np.array([[0, 255, 0]], dtype=np.uint8)
        mp, mc = merge_pointclouds([(p1, c1), (p2, c2)])
        assert mp.shape[0] == 2
        assert mc.shape[0] == 2

    def test_merge_empty(self):
        """Merging empty list returns empty arrays."""
        mp, mc = merge_pointclouds([])
        assert mp.shape[0] == 0


# ── Binary Encoding/Decoding ──

class TestBinaryProtocol:
    def test_roundtrip(self):
        """Encode → decode preserves data."""
        pos = np.array([[1.5, 2.5, 3.5], [-1.0, 0.0, 1.0]], dtype=np.float32)
        col = np.array([[255, 128, 0], [0, 64, 255]], dtype=np.uint8)
        skeleton = {"23": {"x": 0.1, "y": 1.0, "z": 2.0, "confidence": 0.95}}

        data = encode_pointcloud_binary(pos, col, skeleton=skeleton, camera_id=0, timestamp=12345.678)
        result = decode_pointcloud_binary(data)

        assert result["camera_id"] == 0
        assert result["num_points"] == 2
        assert abs(result["timestamp"] - 12345.678) < 0.001
        np.testing.assert_allclose(result["positions"], pos, atol=1e-5)
        np.testing.assert_array_equal(result["colors"], col)
        assert result["skeleton"]["23"]["x"] == 0.1

    def test_roundtrip_no_skeleton(self):
        """Encode → decode works without skeleton."""
        pos = np.array([[0, 0, 0]], dtype=np.float32)
        col = np.array([[128, 128, 128]], dtype=np.uint8)
        data = encode_pointcloud_binary(pos, col, camera_id=-1, timestamp=0.0)
        result = decode_pointcloud_binary(data)
        assert result["camera_id"] == -1
        assert result["skeleton"] is None

    def test_roundtrip_large(self):
        """Encode/decode 50k points without corruption."""
        n = 50000
        pos = np.random.randn(n, 3).astype(np.float32)
        col = np.random.randint(0, 255, (n, 3), dtype=np.uint8)
        data = encode_pointcloud_binary(pos, col, camera_id=1, timestamp=99.9)
        result = decode_pointcloud_binary(data)
        assert result["num_points"] == n
        np.testing.assert_allclose(result["positions"], pos, atol=1e-5)
        np.testing.assert_array_equal(result["colors"], col)

    def test_magic_header(self):
        """Binary starts with correct magic bytes."""
        pos = np.zeros((1, 3), dtype=np.float32)
        col = np.zeros((1, 3), dtype=np.uint8)
        data = encode_pointcloud_binary(pos, col)
        import struct
        magic = struct.unpack_from("<I", data, 0)[0]
        assert magic == 0x50434C44

    def test_bad_magic_raises(self):
        """Decoding bad data raises assertion."""
        with pytest.raises(AssertionError):
            decode_pointcloud_binary(b"\x00\x00\x00\x00" + b"\x00" * 100)


# ── Full E2E: Synthetic Camera → Point Cloud → Encode → Decode ──

class TestE2EVolumetric:
    def test_v2_camera_to_encoded_cloud(self):
        """Simulate v2 camera frame → point cloud → binary encode → decode."""
        info = _make_device_info_v2()
        rgb, depth = _make_rgb_depth(info.width, info.height, 2000.0)

        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=8, max_points=10000)
        assert pos.shape[0] > 0

        # Transform with identity
        world_pos = transform_pointcloud(pos, np.eye(4))

        # Encode
        skeleton = {"23": {"x": 0.1, "y": 1.0, "z": 2.0, "confidence": 0.9}}
        data = encode_pointcloud_binary(world_pos, col, skeleton=skeleton, camera_id=0)

        # Decode
        result = decode_pointcloud_binary(data)
        assert result["num_points"] == pos.shape[0]
        assert result["skeleton"]["23"]["confidence"] == 0.9

    def test_v1_camera_to_encoded_cloud(self):
        """Simulate v1 camera frame → point cloud → binary encode → decode."""
        info = _make_device_info_v1()
        rgb, depth = _make_rgb_depth(info.width, info.height, 1500.0)

        pos, col = rgbd_to_pointcloud(rgb, depth, info, stride=4, max_points=10000)
        assert pos.shape[0] > 0

        data = encode_pointcloud_binary(pos, col, camera_id=1)
        result = decode_pointcloud_binary(data)
        assert result["camera_id"] == 1

    def test_dual_camera_fused_cloud(self):
        """Two cameras → two clouds → merge → encode → decode."""
        info_v1 = _make_device_info_v1()
        info_v2 = _make_device_info_v2()

        rgb1, depth1 = _make_rgb_depth(info_v1.width, info_v1.height, 1800.0)
        rgb2, depth2 = _make_rgb_depth(info_v2.width, info_v2.height, 2200.0)

        pos1, col1 = rgbd_to_pointcloud(rgb1, depth1, info_v1, stride=4, max_points=5000)
        pos2, col2 = rgbd_to_pointcloud(rgb2, depth2, info_v2, stride=8, max_points=5000)

        # Different transforms for each camera
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [2.0, 0, 0]  # camera 2 offset by 2m on X

        wp1 = transform_pointcloud(pos1, T1)
        wp2 = transform_pointcloud(pos2, T2)

        fused_pos, fused_col = merge_pointclouds([(wp1, col1), (wp2, col2)])
        assert fused_pos.shape[0] == pos1.shape[0] + pos2.shape[0]

        # Encode fused
        data = encode_pointcloud_binary(fused_pos, fused_col, camera_id=-1)
        result = decode_pointcloud_binary(data)
        assert result["camera_id"] == -1
        assert result["num_points"] == fused_pos.shape[0]

    def test_alternating_schedule(self):
        """Simulate alternating 15Hz per camera → 30Hz combined."""
        frames_sent = {0: 0, 1: 0}
        num_cameras = 2
        cam_schedule_idx = 0

        for tick in range(60):  # 60 ticks = 2 seconds at 30Hz
            scheduled = cam_schedule_idx % num_cameras
            cam_schedule_idx += 1
            frames_sent[scheduled] += 1

        # Each camera should get ~half the frames
        assert frames_sent[0] == 30
        assert frames_sent[1] == 30

    def test_skeleton_dict_format(self):
        """Skeleton dict uses string joint indices as keys."""
        from camera import JOINT_INDICES
        from fusion import FusedJoint

        joints = {
            "LEFT_HIP": FusedJoint(name="LEFT_HIP", x=0.1, y=1.0, z=2.0, confidence=0.9),
            "RIGHT_HIP": FusedJoint(name="RIGHT_HIP", x=-0.1, y=1.0, z=2.0, confidence=0.85),
            "LEFT_ANKLE": FusedJoint(name="LEFT_ANKLE", x=0.15, y=0.05, z=2.0, confidence=0.1),  # below threshold
        }

        # Import the helper from server
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kinect_server'))
        from server import _build_skeleton_dict

        skel = _build_skeleton_dict(joints)
        assert "23" in skel  # LEFT_HIP index
        assert "24" in skel  # RIGHT_HIP index
        assert "27" not in skel  # LEFT_ANKLE below confidence threshold
        assert skel["23"]["x"] == 0.1
