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


def build_default_calibration(num_cameras: int) -> Dict[int, np.ndarray]:
    """
    Apply evenly-spaced yaw offsets for a horizontal arc arrangement.
    This is a ROUGH default — prints a loud warning.
    """
    logger.warning("=" * 60)
    logger.warning("WARNING: No calibration.json found!")
    logger.warning("Using DEFAULT arc-arrangement calibration.")
    logger.warning("Tracking accuracy will be SIGNIFICANTLY WORSE.")
    logger.warning("Run with --calibrate to perform proper calibration.")
    logger.warning("=" * 60)

    calibration = {}
    if num_cameras == 1:
        calibration[0] = np.eye(4)
    else:
        # Spread cameras in a 180-degree arc
        angle_step = math.pi / (num_cameras - 1)
        radius = 2.0  # assume 2m radius
        for i in range(num_cameras):
            angle = -math.pi / 2 + i * angle_step
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


def run_calibration(cameras, filepath: str = "calibration.json"):
    """
    Interactive checkerboard calibration for multiple cameras.
    Captures 30 frames with all cameras, runs stereo calibration pairwise from camera 0.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION MODE")
    print("=" * 60)
    print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners, {SQUARE_SIZE_MM}mm squares")
    print("Place checkerboard VISIBLE TO ALL CAMERAS simultaneously.")
    print("Starting capture in 5 seconds...")
    print("=" * 60 + "\n")
    time.sleep(5)

    # Collect checkerboard frames
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    all_img_pts = {cam.device_index: [] for cam in cameras}
    all_obj_pts = []

    print(f"Capturing {NUM_FRAMES} frames...")
    frame_idx = 0
    while frame_idx < NUM_FRAMES:
        # Get frame from all cameras
        frames = []
        for cam in cameras:
            f = cam.get_frame(timeout=0.5)
            if f is not None:
                frames.append(f)

        if len(frames) < len(cameras):
            continue

        # Find checkerboard in each camera's preview
        found_all = True
        corners_per_cam = {}
        for frame in frames:
            if frame.rgb_preview is None:
                found_all = False
                break
            gray = cv2.cvtColor(frame.rgb_preview, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (960, 540))  # downsample for speed
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if not ret:
                found_all = False
                break
            corners_per_cam[frame.camera_id] = corners

        if not found_all:
            time.sleep(0.05)
            continue

        all_obj_pts.append(objp)
        for cam_id, corners in corners_per_cam.items():
            all_img_pts[cam_id].append(corners)

        frame_idx += 1
        print(f"  Frame {frame_idx}/{NUM_FRAMES} captured")

    print("Computing calibration matrices...")

    # Camera 0 is the reference (world = camera 0 space for now)
    calibration = {0: np.eye(4)}

    cam0_idx = cameras[0].device_index
    # Image size (downsampled)
    img_size = (960, 540)
    # Approximate intrinsics for downsampled image
    fx = 1081.37 * (960 / 1920)
    fy = 1081.37 * (540 / 1080)
    cx = 959.5 * (960 / 1920)
    cy = 539.5 * (540 / 1080)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    for cam in cameras[1:]:
        cam_idx = cam.device_index
        try:
            ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
                all_obj_pts,
                all_img_pts[cam0_idx],
                all_img_pts[cam_idx],
                K.copy(), dist.copy(),
                K.copy(), dist.copy(),
                img_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            # Transform from cam_idx to cam0 (world)
            mat = np.eye(4)
            mat[:3, :3] = R.T  # invert rotation
            mat[:3, 3] = (-R.T @ T).ravel()
            calibration[cam_idx] = mat
            logger.info(f"Camera {cam_idx}: stereo calibration RMS={ret:.3f}")
        except Exception as e:
            logger.error(f"Camera {cam_idx}: calibration failed: {e}")
            calibration[cam_idx] = np.eye(4)

    save_calibration(calibration, filepath)
    print(f"\nCalibration complete! Saved to {filepath}")
    return calibration
