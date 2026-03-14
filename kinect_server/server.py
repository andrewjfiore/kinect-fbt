#!/usr/bin/env python3
"""
kinect_server/server.py — FBT Linux server entry point.
Orchestrates camera capture, multi-camera fusion, and OSC output.
"""
import argparse
import logging
import sys
import time
from typing import Dict, List

import numpy as np

from camera import KinectCamera, enumerate_kinect_devices
from calibration import build_default_calibration, load_calibration, run_calibration
from fusion import MultiCameraFusion
from osc_output import OSCOutput
from debug_http import init_debug_server, start_debug_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Kinect v2 FBT server for VRChat via OSC")
    p.add_argument("--num-cameras", type=int, default=None,
                   help="Number of Kinect devices (auto-detect if omitted)")
    p.add_argument("--target-ip", type=str, default="192.168.1.100",
                   help="Quest IP address for OSC output")
    p.add_argument("--target-port", type=int, default=39571,
                   help="UDP port to send OSC data to")
    p.add_argument("--fps", type=int, default=20,
                   help="Target FPS for OSC output")
    p.add_argument("--user-height", type=float, default=1.7,
                   help="User height in meters (default 1.7)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run calibration mode and exit")
    p.add_argument("--calibration-file", type=str, default="calibration.json",
                   help="Path to calibration JSON file")
    p.add_argument("--debug-server", action="store_true",
                   help="Enable HTTP debug server on port 8090")
    p.add_argument("--dry-run", action="store_true",
                   help="Print OSC messages to stdout instead of sending UDP")
    return p.parse_args()


def main():
    args = parse_args()

    # Enumerate cameras
    if args.num_cameras is not None:
        num_cameras = args.num_cameras
    else:
        num_cameras = enumerate_kinect_devices()
        if num_cameras == 0:
            logger.error("No Kinect devices found!")
            sys.exit(1)

    logger.info(f"Using {num_cameras} camera(s)")
    cameras = [KinectCamera(i, fps_target=args.fps) for i in range(num_cameras)]

    # Start cameras
    for cam in cameras:
        cam.start()

    # Calibration mode
    if args.calibrate:
        logger.info("Running calibration mode...")
        time.sleep(2)  # Let cameras warm up
        run_calibration(cameras, filepath=args.calibration_file)
        for cam in cameras:
            cam.stop()
        sys.exit(0)

    # Load or build calibration
    calibration = load_calibration(args.calibration_file)
    if calibration is None:
        calibration = build_default_calibration(num_cameras)

    # Init fusion
    fusion = MultiCameraFusion(calibration, user_height=args.user_height)

    # Init OSC
    osc = OSCOutput(args.target_ip, args.target_port, dry_run=args.dry_run)

    # Shared state for debug server
    state: Dict = {
        "cameras": cameras,
        "cameras_active": num_cameras,
        "joints_tracked": 0,
        "fusion_fps": 0.0,
        "osc_target": f"{args.target_ip}:{args.target_port}",
        "joints": {},
        "calibration": calibration,
    }

    # Debug server
    if args.debug_server:
        init_debug_server(state)
        start_debug_server(port=8090)

    frame_interval = 1.0 / args.fps
    fps_counter = 0
    fps_time = time.monotonic()

    logger.info(f"FBT server running at {args.fps}fps → {args.target_ip}:{args.target_port}")
    if args.dry_run:
        logger.info("DRY RUN: OSC messages printed to stdout")

    try:
        while True:
            t_start = time.monotonic()

            # Collect latest frames from all cameras
            frames = []
            for cam in cameras:
                f = cam.get_frame(timeout=frame_interval * 0.5)
                if f is not None:
                    frames.append(f)

            cameras_active = len(frames)
            state["cameras_active"] = cameras_active

            if frames:
                # Fuse joints
                joints = fusion.update(frames)
                state["joints"] = joints
                state["joints_tracked"] = fusion.joints_tracked_count()

                # Get virtual trackers
                trackers = fusion.get_trackers()

                # Send OSC
                osc.send(trackers, cameras_active, fusion.joints_tracked_count(), state["fusion_fps"])

            # FPS tracking
            fps_counter += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                state["fusion_fps"] = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # Sleep to maintain target FPS
            elapsed = time.monotonic() - t_start
            sleep = frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        for cam in cameras:
            cam.stop()


if __name__ == "__main__":
    main()
