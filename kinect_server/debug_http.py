"""
HTTP debug server on port 8090.
Provides JSON status, MJPEG preview streams, and joint/calibration snapshots.
"""
import io
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)
_state: Dict[str, Any] = {}


def init_debug_server(state: Dict[str, Any]):
    """Initialize shared state reference for the debug server."""
    global _state
    _state = state


def start_debug_server(port: int = 8090):
    """Start Flask in a daemon thread."""
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False),
        daemon=True,
        name="debug-http",
    )
    t.start()
    logger.info(f"Debug HTTP server started on port {port}")


@app.route("/")
def status():
    cameras = _state.get("cameras", [])
    cam_info = []
    for c in cameras:
        cam_info.append({
            "id": c.device_index,
            "fps": round(c.fps, 1),
        })
    return jsonify({
        "cameras": cam_info,
        "cameras_active": _state.get("cameras_active", 0),
        "joints_tracked": _state.get("joints_tracked", 0),
        "fusion_fps": round(_state.get("fusion_fps", 0.0), 1),
        "osc_target": _state.get("osc_target", ""),
    })


@app.route("/preview/<int:cam_id>")
def preview(cam_id: int):
    """
    MJPEG preview stream.  Reads from the shared 'preview_frames' dict in state
    rather than consuming frames from the camera queue (which would starve the
    main tracking loop).
    """
    cameras = _state.get("cameras", [])
    cam = next((c for c in cameras if c.device_index == cam_id), None)
    if cam is None:
        return Response("Camera not found", status=404)

    def generate():
        while True:
            # Read from preview buffer (written by main loop, non-destructive)
            preview_frames = _state.get("preview_frames", {})
            frame_data = preview_frames.get(cam_id)
            if frame_data is None or frame_data.get("rgb") is None:
                time.sleep(0.05)
                continue
            rgb = frame_data["rgb"]
            img = cv2.resize(rgb, (960, 540))
            ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(1.0 / 20)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/joints")
def joints():
    joint_data = _state.get("joints", {})
    out = {}
    for name, j in joint_data.items():
        out[name] = {
            "x": round(j.x, 4),
            "y": round(j.y, 4),
            "z": round(j.z, 4),
            "confidence": round(j.confidence, 3),
            "is_lost": j.is_lost,
        }
    return jsonify(out)


@app.route("/calibration")
def calibration():
    cal = _state.get("calibration", {})
    out = {str(k): v.tolist() for k, v in cal.items()}
    return jsonify(out)
