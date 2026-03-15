"""
Multi-camera calibration using OpenCV stereo calibration with a checkerboard.
"""
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 25.0
NUM_FRAMES = 30


def build_default_calibration(num_cameras: int, layout: str = "arc") -> Dict[int, np.ndarray]:
    """
    Build default calibration for a given camera layout.

    Args:
        num_cameras: Number of cameras
        layout: Camera arrangement type
            - "arc": Cameras spread in a 180-degree arc (default for 3+ cameras)
            - "facing": Two cameras on opposite walls facing inward (good for 2 cameras)

    Returns:
        Dict mapping camera_id -> 4x4 transform matrix (camera-to-world)
    """
    logger.warning("=" * 60)
    logger.warning("WARNING: No calibration.json found!")
    logger.warning(f"Using DEFAULT {layout}-arrangement calibration.")
    logger.warning("Tracking accuracy will be SIGNIFICANTLY WORSE.")
    logger.warning("Run with --calibrate to perform proper calibration.")
    logger.warning("=" * 60)

    calibration = {}
    if num_cameras == 1:
        calibration[0] = np.eye(4)

    elif layout == "facing" and num_cameras == 2:
        # Two cameras on opposite walls facing each other
        # Camera 0: at z = -2m, facing +Z (toward origin)
        # Camera 1: at z = +2m, facing -Z (toward origin), rotated 180° around Y
        radius = 2.0

        # Camera 0: identity position at -Z, facing +Z
        calibration[0] = np.eye(4)
        calibration[0][2, 3] = -radius  # position at z = -2m

        # Camera 1: at +Z, rotated 180° around Y (facing -Z)
        yaw = math.pi  # 180 degrees
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy],
        ])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = np.array([0.0, 0.0, radius])  # position at z = +2m
        calibration[1] = mat

    else:
        # Spread cameras in a full 360-degree circle (supports opposing-wall setups)
        angle_step = 2 * math.pi / num_cameras
        radius = 2.0  # assume 2m radius
        for i in range(num_cameras):
            angle = i * angle_step
            # Camera position on arc
            tx = radius * math.cos(angle)
            tz = radius * math.sin(angle)
            # Camera looks toward origin, yaw = angle + 90 degrees
            yaw = angle + math.pi / 2
            cy, sy = math.cos(yaw), math.sin(yaw)
            R = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy],
            ])
            T = np.array([tx, 0.0, tz])
            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3, 3] = T
            calibration[i] = mat

    return calibration


def load_calibration(filepath: str) -> Optional[Dict[int, np.ndarray]]:
    """Load calibration from JSON file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        calibration = {}
        for k, v in data.items():
            calibration[int(k)] = np.array(v)
        logger.info(f"Loaded calibration from {filepath} ({len(calibration)} cameras)")
        return calibration
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")
        return None


def save_calibration(calibration: Dict[int, np.ndarray], filepath: str):
    """Save calibration to JSON file."""
    data = {str(k): v.tolist() for k, v in calibration.items()}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Calibration saved to {filepath}")


def _get_camera_intrinsics_matrix(cam, img_size=(960, 540)):
    """Get intrinsic matrix for a camera, scaled to the given image size."""
    if hasattr(cam, '_intrinsics'):
        intr = cam._intrinsics
        native_w = cam._native_w
        native_h = cam._native_h
    else:
        intr = {"fx": 1081.37, "fy": 1081.37, "cx": 959.5, "cy": 539.5}
        native_w, native_h = 1920, 1080

    scale_x = img_size[0] / native_w
    scale_y = img_size[1] / native_h
    fx = intr["fx"] * scale_x
    fy = intr["fy"] * scale_y
    cx = intr["cx"] * scale_x
    cy = intr["cy"] * scale_y
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _capture_pairwise_frames(cam_a, cam_b, num_frames=NUM_FRAMES):
    """Capture checkerboard frames visible to BOTH cameras in a pair."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_pts = []
    img_pts_a = []
    img_pts_b = []

    print(f"  Capturing {num_frames} frames for cameras {cam_a.device_index} ↔ {cam_b.device_index}...")
    frame_idx = 0
    timeout_start = time.monotonic()
    while frame_idx < num_frames:
        if time.monotonic() - timeout_start > 120:  # 2 min timeout per pair
            logger.warning(f"  Timeout after {frame_idx} frames for pair {cam_a.device_index}-{cam_b.device_index}")
            break

        fa = cam_a.get_frame(timeout=0.5)
        fb = cam_b.get_frame(timeout=0.5)
        if fa is None or fb is None or fa.rgb_preview is None or fb.rgb_preview is None:
            time.sleep(0.05)
            continue

        gray_a = cv2.cvtColor(cv2.resize(fa.rgb_preview, (960, 540)), cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(cv2.resize(fb.rgb_preview, (960, 540)), cv2.COLOR_BGR2GRAY)

        ret_a, corners_a = cv2.findChessboardCorners(gray_a, CHECKERBOARD, None)
        ret_b, corners_b = cv2.findChessboardCorners(gray_b, CHECKERBOARD, None)

        if ret_a and ret_b:
            obj_pts.append(objp)
            img_pts_a.append(corners_a)
            img_pts_b.append(corners_b)
            frame_idx += 1
            print(f"    Frame {frame_idx}/{num_frames}")

        time.sleep(0.05)

    return obj_pts, img_pts_a, img_pts_b


def run_calibration(cameras, filepath: str = "calibration.json"):
    """
    Sequential pairwise checkerboard calibration.
    Calibrates adjacent camera pairs that share a view, then chains transforms.
    Supports opposing cameras that can't see the same checkerboard simultaneously.
    """
    if len(cameras) <= 1:
        calibration = {cameras[0].device_index: np.eye(4)} if cameras else {}
        save_calibration(calibration, filepath)
        return calibration

    print("\n" + "=" * 60)
    print("SEQUENTIAL PAIRWISE CALIBRATION")
    print("=" * 60)
    print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners, {SQUARE_SIZE_MM}mm squares")
    print(f"Calibrating {len(cameras)} cameras in pairs.")
    print("You only need 2 adjacent cameras to see the checkerboard at a time.")
    print("=" * 60 + "\n")

    # Camera 0 is world origin
    calibration = {cameras[0].device_index: np.eye(4)}
    img_size = (960, 540)
    dist = np.zeros(5)

    for i in range(len(cameras) - 1):
        cam_a = cameras[i]
        cam_b = cameras[i + 1]

        print(f"\n--- Pair {i+1}/{len(cameras)-1}: Camera {cam_a.device_index} ↔ Camera {cam_b.device_index} ---")
        print(f"Place checkerboard where BOTH cameras {cam_a.device_index} and {cam_b.device_index} can see it.")
        input("Press Enter when ready...")

        obj_pts, img_pts_a, img_pts_b = _capture_pairwise_frames(cam_a, cam_b)

        if len(obj_pts) < 5:
            logger.error(f"Not enough frames for pair {cam_a.device_index}-{cam_b.device_index}, using identity")
            calibration[cam_b.device_index] = calibration.get(cam_a.device_index, np.eye(4)).copy()
            continue

        K_a = _get_camera_intrinsics_matrix(cam_a, img_size)
        K_b = _get_camera_intrinsics_matrix(cam_b, img_size)

        try:
            ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
                obj_pts, img_pts_a, img_pts_b,
                K_a.copy(), dist.copy(),
                K_b.copy(), dist.copy(),
                img_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            # Transform from cam_b to cam_a
            mat_b_to_a = np.eye(4)
            mat_b_to_a[:3, :3] = R.T
            mat_b_to_a[:3, 3] = (-R.T @ T).ravel()

            # Chain: cam_b→world = cam_a→world @ cam_b→cam_a
            cam_a_to_world = calibration[cam_a.device_index]
            calibration[cam_b.device_index] = cam_a_to_world @ mat_b_to_a
            logger.info(f"Camera {cam_b.device_index}: stereo calibration RMS={ret:.3f}")
            print(f"  Camera {cam_b.device_index} calibrated (RMS={ret:.3f})")
        except Exception as e:
            logger.error(f"Pair {cam_a.device_index}-{cam_b.device_index} calibration failed: {e}")
            calibration[cam_b.device_index] = calibration.get(cam_a.device_index, np.eye(4)).copy()

    save_calibration(calibration, filepath)
    print(f"\nCalibration complete! Saved to {filepath}")
    return calibration


def run_manual_calibration(cameras, filepath: str = "calibration.json"):
    """
    Manual calibration: user enters position (x, y, z in meters) and yaw (degrees) per camera.
    Useful when checkerboard calibration isn't possible (opposing walls, etc.).
    """
    print("\n" + "=" * 60)
    print("MANUAL CALIBRATION")
    print("=" * 60)
    print("Enter position and yaw for each camera.")
    print("Position is in meters from the center of the tracking area.")
    print("Yaw is in degrees (0 = facing +Z, 90 = facing +X, etc.)")
    print("=" * 60 + "\n")

    calibration = {}
    for cam in cameras:
        idx = cam.device_index
        print(f"\nCamera {idx}:")
        try:
            x = float(input(f"  X position (meters, left/right): "))
            y = float(input(f"  Y position (meters, up/down, usually 0): "))
            z = float(input(f"  Z position (meters, forward/back): "))
            yaw_deg = float(input(f"  Yaw (degrees, 0=facing +Z): "))
        except (ValueError, EOFError):
            print(f"  Invalid input, using identity for camera {idx}")
            calibration[idx] = np.eye(4)
            continue

        yaw = math.radians(yaw_deg)
        cy_r, sy_r = math.cos(yaw), math.sin(yaw)
        R = np.array([
            [cy_r, 0, sy_r],
            [0, 1, 0],
            [-sy_r, 0, cy_r],
        ])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = [x, y, z]
        calibration[idx] = mat
        print(f"  Camera {idx}: pos=({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw_deg:.1f}°")

    save_calibration(calibration, filepath)
    print(f"\nManual calibration saved to {filepath}")
    return calibration


def run_tpose_calibration(cameras, filepath: str = "calibration.json"):
    """
    T-pose calibration: user stands in center doing T-pose.
    System detects skeleton from each camera and computes relative transforms
    from the shared skeleton geometry.
    """
    print("\n" + "=" * 60)
    print("T-POSE CALIBRATION")
    print("=" * 60)
    print("Stand in the CENTER of the tracking area in a T-pose.")
    print("  - Arms straight out to the sides")
    print("  - Feet shoulder-width apart")
    print("  - Face any direction (preferably camera 0)")
    print("Starting capture in 5 seconds...")
    print("=" * 60 + "\n")
    time.sleep(5)

    # Key joints for T-pose geometry
    TPOSE_JOINTS = [11, 12, 23, 24, 15, 16]  # shoulders, hips, wrists
    joint_names = {11: "L_SHOULDER", 12: "R_SHOULDER", 23: "L_HIP", 24: "R_HIP",
                   15: "L_WRIST", 16: "R_WRIST"}

    # Collect skeleton observations from each camera
    cam_skeletons = {}  # cam_id -> list of 3D joint positions in camera space
    num_samples = 30

    for cam in cameras:
        cam_id = cam.device_index
        print(f"Capturing from camera {cam_id}...")
        samples = {j: [] for j in TPOSE_JOINTS}
        collected = 0
        timeout_start = time.monotonic()

        while collected < num_samples and time.monotonic() - timeout_start < 30:
            frame = cam.get_frame(timeout=0.5)
            if frame is None:
                continue

            found_all = True
            for j in TPOSE_JOINTS:
                lm = frame.landmarks[j]
                if lm is None or lm.visibility < 0.5:
                    found_all = False
                    break

            if found_all:
                for j in TPOSE_JOINTS:
                    lm = frame.landmarks[j]
                    samples[j].append(np.array([lm.x, lm.y, lm.z]))
                collected += 1

        if collected < 5:
            logger.warning(f"Camera {cam_id}: only got {collected} valid frames, skipping")
            continue

        # Average joint positions for this camera
        avg = {}
        for j in TPOSE_JOINTS:
            avg[j] = np.mean(samples[j], axis=0)
        cam_skeletons[cam_id] = avg
        print(f"  Camera {cam_id}: {collected} frames captured")

    if len(cam_skeletons) < 2:
        logger.error("Need at least 2 cameras with valid skeleton data for T-pose calibration")
        return build_default_calibration(len(cameras))

    # Use camera 0 (or first available) as reference
    ref_cam = cameras[0].device_index
    if ref_cam not in cam_skeletons:
        ref_cam = list(cam_skeletons.keys())[0]

    ref_joints = cam_skeletons[ref_cam]
    calibration = {ref_cam: np.eye(4)}

    # For each other camera, find rigid transform from its skeleton to the reference skeleton
    for cam_id, joints in cam_skeletons.items():
        if cam_id == ref_cam:
            continue

        # Build point clouds from shared joints
        src_pts = np.array([joints[j] for j in TPOSE_JOINTS])
        dst_pts = np.array([ref_joints[j] for j in TPOSE_JOINTS])

        # Solve rigid body transform using SVD (Kabsch algorithm)
        src_centroid = np.mean(src_pts, axis=0)
        dst_centroid = np.mean(dst_pts, axis=0)
        src_centered = src_pts - src_centroid
        dst_centered = dst_pts - dst_centroid

        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1, not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_centroid - R @ src_centroid

        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t
        calibration[cam_id] = mat

        # Compute residual error
        transformed = (R @ src_pts.T).T + t
        error = np.mean(np.linalg.norm(transformed - dst_pts, axis=1))
        print(f"  Camera {cam_id}: T-pose calibration error = {error*1000:.1f}mm")
        logger.info(f"Camera {cam_id}: T-pose calibration, mean error={error*1000:.1f}mm")

    # Fill in any cameras that didn't get skeleton data
    for cam in cameras:
        if cam.device_index not in calibration:
            logger.warning(f"Camera {cam.device_index}: no skeleton data, using identity")
            calibration[cam.device_index] = np.eye(4)

    save_calibration(calibration, filepath)
    print(f"\nT-pose calibration complete! Saved to {filepath}")
    return calibration
