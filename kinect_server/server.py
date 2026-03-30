#!/usr/bin/env python3
"""
kinect_server/server.py — FBT server entry point.
Orchestrates camera capture, multi-camera fusion, and OSC output.
Supports Kinect v1 (Xbox 360) and v2 (Xbox One), including mixed setups.
Optional volumetric web server for 3D point cloud monitoring.
"""
import os as _os
# Suppress TFLite C++ warnings from MediaPipe's internal inference engine
# (e.g. "Feedback manager requires a model with a single signature inference")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import logging
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from camera import KinectCamera, CameraFrame, enumerate_kinect_devices
from calibration import build_default_calibration, load_calibration, run_calibration, run_manual_calibration, run_tpose_calibration
from fusion import MultiCameraFusion, FusedJoint
from osc_output import OSCOutput
from debug_http import init_debug_server, start_debug_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Frame cache staleness TTL (seconds)
FRAME_CACHE_TTL = 0.15  # 150ms — use cached frame if fresh enough


def parse_args():
    p = argparse.ArgumentParser(description="Kinect FBT server for VRChat via OSC")
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
    p.add_argument("--layout", type=str, default="auto", choices=["arc", "facing", "auto"],
                   help="Camera layout for default calibration: 'arc' (semicircle), "
                        "'facing' (2 cameras on opposite walls), or 'auto' (facing if 2 cams, arc otherwise)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run checkerboard calibration mode and exit")
    p.add_argument("--calibrate-manual", action="store_true",
                   help="Run manual calibration (enter position/yaw per camera)")
    p.add_argument("--calibrate-tpose", action="store_true",
                   help="Run T-pose calibration (stand in center)")
    p.add_argument("--calibration-file", type=str, default="calibration.json",
                   help="Path to calibration JSON file")
    p.add_argument("--debug-server", action="store_true",
                   help="Enable HTTP debug server on port 8090")
    p.add_argument("--web-server", action="store_true",
                   help="Enable volumetric 3D web server (WebSocket + Three.js) on port 8090")
    p.add_argument("--web-port", type=int, default=8090,
                   help="Port for volumetric web server (default 8090)")
    p.add_argument("--web-stride", type=int, default=4,
                   help="Point cloud downsample stride (default 4; higher = fewer points, faster)")
    p.add_argument("--web-max-points", type=int, default=50000,
                   help="Max points per camera per frame (default 50000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print OSC messages to stdout instead of sending UDP")
    return p.parse_args()


def _build_skeleton_dict(joints: Dict[str, FusedJoint]) -> dict:
    """Convert fused joints to a compact dict for WebSocket streaming."""
    out = {}
    for name, j in joints.items():
        if j.is_lost or j.confidence < 0.2:
            continue
        # Use MediaPipe landmark index as key (matches JOINT_INDICES)
        from camera import JOINT_INDICES
        idx = JOINT_INDICES.get(name)
        if idx is not None:
            out[str(idx)] = {
                "x": round(j.x, 4),
                "y": round(j.y, 4),
                "z": round(j.z, 4),
                "confidence": round(j.confidence, 3),
            }
    return out


def main():
    args = parse_args()

    # Enumerate cameras
    if args.num_cameras is not None:
        num_cameras = args.num_cameras
        if num_cameras <= 0:
            logger.error("--num-cameras must be >= 1")
            sys.exit(1)
    else:
        num_cameras = enumerate_kinect_devices()
        if num_cameras == 0:
            logger.error("No Kinect devices found!")
            sys.exit(1)

    # Detect device types for mixed v1/v2 setups
    from camera import enumerate_kinect_devices_detailed
    detected = enumerate_kinect_devices_detailed()

    logger.info(f"Using {num_cameras} camera(s)")
    cameras = []
    for i in range(num_cameras):
        if i < len(detected):
            dev = detected[i]
            cameras.append(KinectCamera(i, fps_target=args.fps,
                                         intrinsics=dev.intrinsics,
                                         device_type=dev.device_type))
        else:
            cameras.append(KinectCamera(i, fps_target=args.fps))

    # Start cameras
    for cam in cameras:
        cam.start()

    # Calibration modes
    if args.calibrate:
        logger.info("Running checkerboard calibration mode...")
        time.sleep(2)
        run_calibration(cameras, filepath=args.calibration_file)
        for cam in cameras:
            cam.stop()
        sys.exit(0)

    if args.calibrate_manual:
        logger.info("Running manual calibration mode...")
        run_manual_calibration(cameras, filepath=args.calibration_file)
        for cam in cameras:
            cam.stop()
        sys.exit(0)

    if args.calibrate_tpose:
        logger.info("Running T-pose calibration mode...")
        time.sleep(2)
        run_tpose_calibration(cameras, filepath=args.calibration_file)
        for cam in cameras:
            cam.stop()
        sys.exit(0)

    # Determine layout for default calibration
    layout = args.layout
    if layout == "auto":
        layout = "facing" if num_cameras == 2 else "arc"

    # Load or build calibration
    calibration = load_calibration(args.calibration_file)
    if calibration is None:
        calibration = build_default_calibration(num_cameras, layout=layout)

    # Init fusion
    fusion = MultiCameraFusion(calibration, user_height=args.user_height)

    # Init OSC
    osc = OSCOutput(args.target_ip, args.target_port, dry_run=args.dry_run)

    # Frame cache: store last frame per camera with timestamp for staleness TTL
    frame_cache: Dict[int, CameraFrame] = {}

    # Shared state for debug server
    state: Dict = {
        "cameras": cameras,
        "cameras_active": num_cameras,
        "joints_tracked": 0,
        "fusion_fps": 0.0,
        "osc_target": f"{args.target_ip}:{args.target_port}",
        "joints": {},
        "calibration": calibration,
        # preview_frames: cam_id -> {"rgb": np.ndarray}  (non-destructive, for HTTP debug)
        "preview_frames": {},
    }

    # Debug server (legacy Flask)
    if args.debug_server and not args.web_server:
        init_debug_server(state)
        start_debug_server(port=8090)

    # Volumetric web server (replaces debug server when enabled)
    web_server = None
    if args.web_server:
        from web_server import VolumetricWebServer
        from pointcloud import rgbd_to_pointcloud, transform_pointcloud, merge_pointclouds, encode_pointcloud_binary
        web_server = VolumetricWebServer(host="0.0.0.0", port=args.web_port)
        web_server.start()
        logger.info(f"Volumetric web server on http://0.0.0.0:{args.web_port}")

    frame_interval = 1.0 / args.fps
    fps_counter = 0
    fps_time = time.monotonic()

    # Alternating capture: each camera gets 15Hz, combined 30Hz
    # cam_schedule tracks which camera to prioritize for point cloud generation
    cam_schedule_idx = 0
    status_broadcast_time = time.monotonic()
    pc_debug_time = time.monotonic()
    pc_debug_count = 0

    logger.info(f"FBT server running at {args.fps}fps → {args.target_ip}:{args.target_port}")
    logger.info(f"Layout: {layout}")
    if args.dry_run:
        logger.info("DRY RUN: OSC messages printed to stdout")
    if web_server:
        logger.info(f"Volumetric streaming: stride={args.web_stride}, max_points={args.web_max_points}")

    try:
        while True:
            t_start = time.monotonic()
            now = time.monotonic()

            # Collect latest frames from all cameras, using cache if unavailable
            frames_to_fuse: List[CameraFrame] = []
            for cam in cameras:
                f = cam.get_frame(timeout=frame_interval * 0.5)
                if f is not None:
                    # Update cache with fresh frame
                    frame_cache[cam.device_index] = f
                    frames_to_fuse.append(f)
                elif cam.device_index in frame_cache:
                    # Use cached frame if not too stale
                    cached = frame_cache[cam.device_index]
                    age = now - cached.timestamp
                    if age < FRAME_CACHE_TTL:
                        frames_to_fuse.append(cached)

            cameras_active = len(frames_to_fuse)
            state["cameras_active"] = cameras_active

            if frames_to_fuse:
                # Fuse joints
                joints = fusion.update(frames_to_fuse)
                state["joints"] = joints
                state["joints_tracked"] = fusion.joints_tracked_count()

                # Update preview frames for HTTP debug server (non-destructive copy)
                for f in frames_to_fuse:
                    if f.rgb_preview is not None:
                        state["preview_frames"][f.camera_id] = {"rgb": f.rgb_preview}

                # Get virtual trackers
                trackers = fusion.get_trackers()

                # Send OSC
                osc.send(trackers, cameras_active, fusion.joints_tracked_count(), state["fusion_fps"])

                # ── Volumetric streaming ──
                if web_server and web_server.client_count > 0:
                    skeleton_dict = _build_skeleton_dict(joints)

                    # Alternating 15Hz: pick which camera sends this tick
                    scheduled_cam = cam_schedule_idx % num_cameras
                    cam_schedule_idx += 1

                    per_cam_clouds = []

                    for f in frames_to_fuse:
                        if f.depth_frame is None or f.rgb_preview is None:
                            continue
                        if f.device_info is None:
                            continue

                        # Use rgb_clean (before skeleton overlay) if available,
                        # otherwise copy rgb_preview to avoid skeleton circles in point cloud
                        rgb_for_pc = getattr(f, 'rgb_clean', None)
                        if rgb_for_pc is None:
                            rgb_for_pc = f.rgb_preview

                        # Generate point cloud for this camera
                        positions, colors = rgbd_to_pointcloud(
                            rgb_for_pc, f.depth_frame, f.device_info,
                            stride=args.web_stride,
                            max_points=args.web_max_points,
                        )

                        if positions.shape[0] == 0:
                            continue

                        # Transform to world space using calibration
                        cam_transform = calibration.get(f.camera_id, np.eye(4))
                        world_positions = transform_pointcloud(positions, cam_transform)

                        # Send per-camera cloud (only for the scheduled camera this tick)
                        if f.camera_id == scheduled_cam:
                            frame_data = encode_pointcloud_binary(
                                world_positions, colors,
                                skeleton=skeleton_dict,
                                camera_id=f.camera_id,
                                timestamp=f.timestamp,
                            )
                            web_server.broadcast_pointcloud(frame_data)

                        per_cam_clouds.append((world_positions, colors))

                    # Debug: log point cloud generation rate
                    pc_debug_count += 1
                    if now - pc_debug_time >= 5.0:
                        logger.info(f"[PC] {pc_debug_count} clouds in 5s, {len(per_cam_clouds)} cams, clients={web_server.client_count}")
                        if per_cam_clouds:
                            logger.info(f"[PC] cloud0: {per_cam_clouds[0][0].shape[0]} pts")
                        pc_debug_count = 0
                        pc_debug_time = now

                    # Send fused cloud (every tick, combining all cameras)
                    if per_cam_clouds:
                        fused_pos, fused_col = merge_pointclouds(per_cam_clouds)
                        # Cap total fused points
                        if fused_pos.shape[0] > args.web_max_points:
                            idx = np.random.choice(fused_pos.shape[0], args.web_max_points, replace=False)
                            fused_pos = fused_pos[idx]
                            fused_col = fused_col[idx]
                        fused_data = encode_pointcloud_binary(
                            fused_pos, fused_col,
                            skeleton=skeleton_dict,
                            camera_id=-1,
                            timestamp=time.monotonic(),
                        )
                        web_server.broadcast_pointcloud(fused_data)

                    # Periodic status broadcast (1Hz)
                    if now - status_broadcast_time >= 1.0:
                        web_server.broadcast_status({
                            "cameras_active": cameras_active,
                            "joints_tracked": fusion.joints_tracked_count(),
                            "fusion_fps": round(state["fusion_fps"], 1),
                            "osc_target": state["osc_target"],
                        })
                        web_server.update_status(
                            cameras_active=cameras_active,
                            joints_tracked=fusion.joints_tracked_count(),
                            fusion_fps=round(state["fusion_fps"], 1),
                        )
                        status_broadcast_time = now

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
        if web_server:
            web_server.stop()


if __name__ == "__main__":
    main()
