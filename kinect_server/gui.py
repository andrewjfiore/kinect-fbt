"""
kinect_server/gui.py — Cross-platform tkinter GUI for the FBT server.
Works on Windows and Linux. Bundles into a single executable via PyInstaller.

Features:
- One-click Start/Stop with live status dashboard
- Live camera feed previews with skeleton overlay
- Multi-body detection and selection
- Variable capture rate slider (10-60 Hz)

References: SlimeVR Server GUI, KinectToVR (K2EX) UI patterns.
"""
import os as _os
# Suppress TFLite C++ warnings from MediaPipe's internal inference engine
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import platform
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

IS_WINDOWS = platform.system() == "Windows"

# ── Logging bridge to GUI ────────────────────────────────────────────────────
class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        try:
            self.log_queue.put_nowait(msg)
        except queue.Full:
            pass


# ── Colors ───────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
BG2     = "#2a2a3d"
BG3     = "#1a1a2a"
ACCENT  = "#7c3aed"  # purple
GREEN   = "#22c55e"
YELLOW  = "#eab308"
RED     = "#ef4444"
CYAN    = "#06b6d4"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
FONT    = ("Segoe UI" if IS_WINDOWS else "Ubuntu", 10)
FONT_B  = (FONT[0], 10, "bold")
FONT_H  = (FONT[0], 13, "bold")
FONT_SM = (FONT[0], 8)

# ── Skeleton limb connections ────────────────────────────────────────────────
LIMB_CONNECTIONS = [
    # Torso
    (11, 12),  # L shoulder - R shoulder
    (11, 23),  # L shoulder - L hip
    (12, 24),  # R shoulder - R hip
    (23, 24),  # L hip - R hip
    # Left arm
    (11, 13),  # L shoulder - L elbow
    (13, 15),  # L elbow - L wrist
    # Right arm
    (12, 14),  # R shoulder - R elbow
    (14, 16),  # R elbow - R wrist
    # Left leg
    (23, 25),  # L hip - L knee
    (25, 27),  # L knee - L ankle
    (27, 29),  # L ankle - L heel
    (27, 31),  # L ankle - L foot index
    # Right leg
    (24, 26),  # R hip - R knee
    (26, 28),  # R knee - R ankle
    (28, 30),  # R ankle - R heel
    (28, 32),  # R ankle - R foot index
    # Head
    (0, 11),   # Nose - L shoulder (approximation)
    (0, 12),   # Nose - R shoulder
]

PREVIEW_W = 480
PREVIEW_H = 270
PREVIEW_FPS = 15


class StatusDot(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=14, height=14, bg=BG2,
                         highlightthickness=0, **kw)
        self._dot = self.create_oval(2, 2, 12, 12, fill=RED, outline="")

    def set(self, color: str):
        self.itemconfig(self._dot, fill=color)


class ConfBar(tk.Canvas):
    def __init__(self, parent, width=120, **kw):
        super().__init__(parent, width=width, height=10, bg=BG2,
                         highlightthickness=0, **kw)
        self._bar = self.create_rectangle(0, 0, 0, 10, fill=GREEN, outline="")
        self._w = width

    def set(self, value: float):
        self.coords(self._bar, 0, 0, int(self._w * max(0, min(1, value))), 10)


# ── Camera Feed Panel ────────────────────────────────────────────────────────
class CameraFeedPanel(tk.LabelFrame):
    """Live camera feed preview with skeleton overlay and body selection."""

    def __init__(self, parent, cam_id: int, **kw):
        super().__init__(parent, text=f"Camera {cam_id}", bg=BG2, fg=TEXT,
                         font=FONT_B, relief="flat", padx=4, pady=4, **kw)
        self.cam_id = cam_id
        self._photo = None
        self._bodies: List[dict] = []  # detected bodies with bounding boxes
        self._selected_body: int = 0   # index of selected body

        # Top info bar
        info = tk.Frame(self, bg=BG2)
        info.pack(fill="x", pady=(0, 2))
        self.lbl_cam_id = tk.Label(info, text=f"Cam {cam_id}", font=FONT_SM,
                                   bg=BG2, fg=CYAN, anchor="w")
        self.lbl_cam_id.pack(side="left")
        self.lbl_fps = tk.Label(info, text="0 FPS", font=FONT_SM,
                                bg=BG2, fg=SUBTEXT, anchor="e")
        self.lbl_fps.pack(side="right")
        self.lbl_body = tk.Label(info, text="", font=FONT_SM,
                                 bg=BG2, fg=YELLOW, anchor="e")
        self.lbl_body.pack(side="right", padx=(0, 8))

        # Body selector dropdown
        self.var_body = tk.StringVar(value="Auto (nearest)")
        self.body_menu = ttk.Combobox(info, textvariable=self.var_body,
                                      values=["Auto (nearest)"], state="readonly",
                                      width=16)
        self.body_menu.pack(side="right", padx=(0, 4))
        self.body_menu.bind("<<ComboboxSelected>>", self._on_body_select)

        # Canvas for video feed
        self.canvas = tk.Canvas(self, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#0a0a14", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Placeholder text
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                                text="No feed", fill=SUBTEXT, font=FONT,
                                tags="placeholder")

    def update_feed(self, rgb_frame: Optional[np.ndarray],
                    landmarks: Optional[list] = None,
                    bodies: Optional[List[dict]] = None,
                    fps: float = 0.0):
        """Update the camera feed display with optional skeleton overlay."""
        if Image is None or ImageTk is None:
            return

        self.lbl_fps.config(text=f"{fps:.0f} FPS")

        if rgb_frame is None:
            return

        self.canvas.delete("placeholder")

        # Resize to preview dimensions
        img = cv2.resize(rgb_frame, (PREVIEW_W, PREVIEW_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb_frame.shape[:2]
        sx = PREVIEW_W / w_orig
        sy = PREVIEW_H / h_orig

        # Draw body bounding boxes
        if bodies:
            self._bodies = bodies
            body_labels = ["Auto (nearest)"]
            for i, body in enumerate(bodies):
                selected = (i == self._selected_body)
                bbox = body.get("bbox")  # (x1, y1, x2, y2) in original coords
                if bbox:
                    x1 = int(bbox[0] * sx)
                    y1 = int(bbox[1] * sy)
                    x2 = int(bbox[2] * sx)
                    y2 = int(bbox[3] * sy)
                    color = (0, 255, 0) if selected else (100, 100, 255)
                    thickness = 2 if selected else 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    label = f"Body {i + 1}"
                    if selected:
                        label += " (selected)"
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    body_labels.append(f"Body {i + 1}")

            self.body_menu["values"] = body_labels
            self.lbl_body.config(text=f"{len(bodies)} bodies")
        else:
            self._bodies = []
            self.lbl_body.config(text="")

        # Draw skeleton overlay
        if landmarks:
            self._draw_skeleton_overlay(img, landmarks, sx, sy, w_orig, h_orig)

        # Convert to PhotoImage
        pil_img = Image.fromarray(img)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _draw_skeleton_overlay(self, img: np.ndarray, landmarks: list,
                               sx: float, sy: float,
                               w_orig: int, h_orig: int):
        """Draw skeleton joints and limb connections on the preview image."""
        # Build pixel position map for drawing limbs
        joint_pixels = {}

        from camera import DEFAULT_V2_INTRINSICS  # import once, outside loop
        _fx = DEFAULT_V2_INTRINSICS["fx"]
        _fy = DEFAULT_V2_INTRINSICS["fy"]
        _cx = DEFAULT_V2_INTRINSICS["cx"]
        _cy = DEFAULT_V2_INTRINSICS["cy"]

        for lm in landmarks:
            if lm is None:
                continue
            idx = lm.index
            # Re-project 3D back to 2D pixel coords for display
            if hasattr(lm, '_px') and hasattr(lm, '_py'):
                px = int(lm._px * sx)
                py = int(lm._py * sy)
            else:
                # Approximate: use x/z and y/z projection
                if lm.z > 0.01:
                    # Rough reprojection using default v2 intrinsics (scaled to orig res)
                    px_orig = int(lm.x * _fx / lm.z + _cx)
                    py_orig = int(lm.y * _fy / lm.z + _cy)
                    px = int(px_orig * sx)
                    py = int(py_orig * sy)
                else:
                    continue

            px = max(0, min(px, PREVIEW_W - 1))
            py = max(0, min(py, PREVIEW_H - 1))
            joint_pixels[idx] = (px, py)

            # Color by confidence
            conf = lm.depth_confidence
            if conf >= 0.7:
                color = (0, 255, 0)    # green = high
            elif conf >= 0.3:
                color = (0, 255, 255)  # yellow = medium
            else:
                color = (0, 0, 255)    # red = low

            cv2.circle(img, (px, py), 4, color, -1)
            cv2.circle(img, (px, py), 5, color, 1)

        # Draw limb connections
        for (a, b) in LIMB_CONNECTIONS:
            if a in joint_pixels and b in joint_pixels:
                pa = joint_pixels[a]
                pb = joint_pixels[b]
                cv2.line(img, pa, pb, (200, 200, 200), 1, cv2.LINE_AA)

    def _on_body_select(self, event):
        val = self.var_body.get()
        if val == "Auto (nearest)":
            self._selected_body = -1  # auto
        else:
            try:
                idx = int(val.split(" ")[1]) - 1
                self._selected_body = idx
            except (ValueError, IndexError):
                self._selected_body = -1

    def _on_canvas_click(self, event):
        """Click on a bounding box to select that body."""
        if not self._bodies:
            return
        # Check if click is inside any bounding box
        for i, body in enumerate(self._bodies):
            bbox = body.get("bbox")
            if not bbox:
                continue
            # Scale bbox to preview coords
            h_orig = body.get("frame_h", 1080)
            w_orig = body.get("frame_w", 1920)
            sx = PREVIEW_W / w_orig
            sy = PREVIEW_H / h_orig
            x1 = int(bbox[0] * sx)
            y1 = int(bbox[1] * sy)
            x2 = int(bbox[2] * sx)
            y2 = int(bbox[3] * sy)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._selected_body = i
                self.var_body.set(f"Body {i + 1}")
                return

    @property
    def selected_body_index(self) -> int:
        return self._selected_body


# ── Person Detector ──────────────────────────────────────────────────────────
class PersonDetector:
    """Lightweight person detector using OpenCV HOG for multi-body bounding boxes."""

    def __init__(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, rgb_frame: np.ndarray) -> List[dict]:
        """Detect people in frame, return list of {bbox, center, area}."""
        h, w = rgb_frame.shape[:2]
        # Downscale for speed
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            small = cv2.resize(rgb_frame, (640, int(h * scale)))
        else:
            small = rgb_frame

        boxes, weights = self._hog.detectMultiScale(
            small, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        bodies = []
        for i, (bx, by, bw, bh) in enumerate(boxes):
            # Scale back to original coords
            x1 = int(bx / scale)
            y1 = int(by / scale)
            x2 = int((bx + bw) / scale)
            y2 = int((by + bh) / scale)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bodies.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "area": (x2 - x1) * (y2 - y1),
                "frame_w": w,
                "frame_h": h,
                "weight": float(weights[i]) if i < len(weights) else 0.0,
            })

        # Sort by distance to frame center (for auto-select)
        frame_cx, frame_cy = w / 2, h / 2
        bodies.sort(key=lambda b: (b["center"][0] - frame_cx) ** 2 +
                                   (b["center"][1] - frame_cy) ** 2)
        return bodies

    @staticmethod
    def select_nearest_center(bodies: List[dict], frame_w: int, frame_h: int) -> int:
        """Return index of body closest to frame center."""
        if not bodies:
            return 0
        cx, cy = frame_w / 2, frame_h / 2
        best = 0
        best_dist = float("inf")
        for i, b in enumerate(bodies):
            dx = b["center"][0] - cx
            dy = b["center"][1] - cy
            d = dx * dx + dy * dy
            if d < best_dist:
                best_dist = d
                best = i
        return best


# ── Main GUI ─────────────────────────────────────────────────────────────────
class FBTServerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FBT Server — Kinect v2 Full-Body Tracking")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(900, 700)

        # Set window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except Exception:
            pass

        self._server_thread: threading.Thread = None
        self._server_stop_event = threading.Event()
        self._log_queue: queue.Queue = queue.Queue(maxsize=200)
        self._status: dict = {}
        self._running = False

        # Shared state for live preview
        self._preview_frames: Dict[int, dict] = {}  # cam_id -> {rgb, landmarks, bodies, fps}
        self._preview_lock = threading.Lock()
        self._feed_panels: Dict[int, CameraFeedPanel] = {}
        self._person_detector = PersonDetector()

        # Variable capture rate (shared with server thread)
        self._target_fps = tk.IntVar(value=20)
        self._frame_interval_lock = threading.Lock()
        self._current_frame_interval = 1.0 / 20

        self._build_ui()
        self._setup_logging()
        self._poll()
        self._poll_preview()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_logging(self):
        handler = QueueHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                               datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _build_ui(self):
        root = self.root

        # ── Header ──────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🦴  FBT Server", font=(FONT[0], 16, "bold"),
                 bg=ACCENT, fg=TEXT).pack(side="left", padx=16)
        tk.Label(hdr, text="Kinect v2 → VRChat Full-Body Tracking",
                 font=FONT, bg=ACCENT, fg="#d4b8ff").pack(side="left")

        # ── Main layout: left (controls) + center (feeds) + right (log) ──
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(main, bg=BG, width=320)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(main, bg=BG, width=360)
        right.pack(side="left", fill="both", expand=True)

        # ── Settings card ───────────────────────────────────────────────────
        self._settings_card(left)

        # ── Status card ─────────────────────────────────────────────────────
        self._status_card(left)

        # ── Start / Stop button ─────────────────────────────────────────────
        self.btn_start = tk.Button(
            left, text="▶  Start Server",
            font=(FONT[0], 12, "bold"),
            bg=GREEN, fg="white", activebackground="#16a34a",
            relief="flat", pady=10, cursor="hand2",
            command=self._toggle_server,
        )
        self.btn_start.pack(fill="x", pady=(8, 4))

        self.btn_calibrate = tk.Button(
            left, text="🎯  Run Calibration",
            font=FONT, bg=BG2, fg=TEXT, relief="flat", pady=6,
            cursor="hand2", command=self._run_calibration,
        )
        self.btn_calibrate.pack(fill="x", pady=(0, 4))

        # ── Camera feeds area ───────────────────────────────────────────────
        feed_label = tk.Label(center, text="Camera Feeds", font=FONT_H, bg=BG, fg=TEXT)
        feed_label.pack(anchor="w")

        self.feed_container = tk.Frame(center, bg=BG)
        self.feed_container.pack(fill="both", expand=True)

        # Placeholder
        self.feed_placeholder = tk.Label(
            self.feed_container,
            text="Start server to see camera feeds",
            font=FONT, bg=BG3, fg=SUBTEXT, pady=80
        )
        self.feed_placeholder.pack(fill="both", expand=True)

        # ── Log view ────────────────────────────────────────────────────────
        tk.Label(right, text="Server Log", font=FONT_H, bg=BG, fg=TEXT).pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(
            right, bg="#0f0f1a", fg=TEXT, font=("Consolas" if IS_WINDOWS else "Monospace", 9),
            relief="flat", state="disabled", wrap="word",
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.tag_config("ERROR",   foreground=RED)
        self.log_box.tag_config("WARNING", foreground=YELLOW)
        self.log_box.tag_config("INFO",    foreground=TEXT)

        btn_clear = tk.Button(right, text="Clear Log", font=FONT_SM, bg=BG2, fg=SUBTEXT,
                              relief="flat", command=self._clear_log)
        btn_clear.pack(anchor="e", pady=4)

        # ── Status bar ──────────────────────────────────────────────────────
        bar = tk.Frame(root, bg=BG2, pady=3)
        bar.pack(fill="x", side="bottom")
        self.lbl_statusbar = tk.Label(bar, text="Ready", font=FONT_SM, bg=BG2, fg=SUBTEXT)
        self.lbl_statusbar.pack(side="left", padx=8)
        tk_os = tk.Label(bar, text=f"OS: {platform.system()} {platform.release()}",
                         font=FONT_SM, bg=BG2, fg=SUBTEXT)
        tk_os.pack(side="right", padx=8)

    def _settings_card(self, parent):
        card = tk.LabelFrame(parent, text="Settings", bg=BG2, fg=TEXT,
                              font=FONT_B, relief="flat", padx=8, pady=6)
        card.pack(fill="x", pady=(0, 8))

        def row(label, default):
            f = tk.Frame(card, bg=BG2)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=label, font=FONT, bg=BG2, fg=SUBTEXT, width=16, anchor="w").pack(side="left")
            e = tk.Entry(f, font=FONT, bg="#12121f", fg=TEXT, insertbackground=TEXT,
                         relief="flat", bd=4)
            e.insert(0, default)
            e.pack(side="left", fill="x", expand=True)
            return e

        self.e_target_ip   = row("Quest IP",        "192.168.1.100")
        self.e_target_port = row("OSC Port",         "39571")
        self.e_height      = row("User Height (m)",  "1.70")
        self.e_cal_file    = row("Calibration File", "calibration.json")

        # Capture Rate slider (replaces FPS text entry)
        f_fps = tk.Frame(card, bg=BG2)
        f_fps.pack(fill="x", pady=2)
        tk.Label(f_fps, text="Capture Rate", font=FONT, bg=BG2, fg=SUBTEXT,
                 width=16, anchor="w").pack(side="left")
        self.slider_fps = tk.Scale(
            f_fps, from_=10, to=60, orient="horizontal",
            variable=self._target_fps,
            bg=BG2, fg=TEXT, troughcolor="#12121f",
            highlightthickness=0, sliderrelief="flat",
            font=FONT_SM, length=140,
            command=self._on_fps_change,
        )
        self.slider_fps.pack(side="left", fill="x", expand=True)
        self.lbl_hz = tk.Label(f_fps, text="20 Hz", font=FONT_SM, bg=BG2, fg=CYAN, width=6)
        self.lbl_hz.pack(side="left")

        # Camera count
        f = tk.Frame(card, bg=BG2)
        f.pack(fill="x", pady=2)
        tk.Label(f, text="Cameras", font=FONT, bg=BG2, fg=SUBTEXT, width=16, anchor="w").pack(side="left")
        self.var_cams = tk.StringVar(value="auto")
        tk.OptionMenu(f, self.var_cams, "auto", "1", "2", "3", "4").pack(side="left")

        # Dry run
        self.var_dryrun = tk.BooleanVar(value=False)
        tk.Checkbutton(card, text="Dry Run (print OSC, no UDP)", variable=self.var_dryrun,
                       bg=BG2, fg=TEXT, selectcolor=BG, activebackground=BG2,
                       font=FONT).pack(anchor="w")

        # Debug server
        self.var_debug = tk.BooleanVar(value=True)
        tk.Checkbutton(card, text="HTTP Debug Server (port 8090)", variable=self.var_debug,
                       bg=BG2, fg=TEXT, selectcolor=BG, activebackground=BG2,
                       font=FONT).pack(anchor="w")

    def _on_fps_change(self, val):
        fps = int(val)
        self.lbl_hz.config(text=f"{fps} Hz")
        with self._frame_interval_lock:
            self._current_frame_interval = 1.0 / fps

    def _status_card(self, parent):
        card = tk.LabelFrame(parent, text="Status", bg=BG2, fg=TEXT,
                              font=FONT_B, relief="flat", padx=8, pady=6)
        card.pack(fill="x", pady=(0, 8))

        def row(label):
            f = tk.Frame(card, bg=BG2)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=label, font=FONT, bg=BG2, fg=SUBTEXT, width=16, anchor="w").pack(side="left")
            lbl = tk.Label(f, text="—", font=FONT, bg=BG2, fg=TEXT, anchor="w")
            lbl.pack(side="left")
            return lbl

        # Server running dot
        f = tk.Frame(card, bg=BG2)
        f.pack(fill="x", pady=2)
        tk.Label(f, text="Server", font=FONT, bg=BG2, fg=SUBTEXT, width=16, anchor="w").pack(side="left")
        self.dot_server = StatusDot(f)
        self.dot_server.pack(side="left", padx=(0, 6))
        self.lbl_server_state = tk.Label(f, text="Stopped", font=FONT, bg=BG2, fg=RED, anchor="w")
        self.lbl_server_state.pack(side="left")

        self.lbl_cameras   = row("Cameras Active")
        self.lbl_joints    = row("Joints Tracked")
        self.lbl_osc       = row("OSC Target")

        # FPS: actual vs target
        f_fps = tk.Frame(card, bg=BG2)
        f_fps.pack(fill="x", pady=2)
        tk.Label(f_fps, text="Fusion FPS", font=FONT, bg=BG2, fg=SUBTEXT,
                 width=16, anchor="w").pack(side="left")
        self.lbl_fps_actual = tk.Label(f_fps, text="—", font=FONT, bg=BG2, fg=TEXT, anchor="w")
        self.lbl_fps_actual.pack(side="left")
        self.lbl_fps_target = tk.Label(f_fps, text="", font=FONT_SM, bg=BG2, fg=SUBTEXT, anchor="w")
        self.lbl_fps_target.pack(side="left", padx=(4, 0))

        # Tracker confidence bars
        tk.Label(card, text="Trackers", font=FONT, bg=BG2, fg=SUBTEXT).pack(anchor="w", pady=(4, 0))
        self.tracker_frame = tk.Frame(card, bg=BG2)
        self.tracker_frame.pack(fill="x")
        self._tracker_bars: dict = {}

    def _update_tracker_ui(self, tracker_states: dict):
        for tid in range(1, 6):
            t = tracker_states.get(tid)
            if t is None:
                if tid in self._tracker_bars:
                    self._tracker_bars[tid][0].config(fg=SUBTEXT)
                    self._tracker_bars[tid][1].set(0)
                continue
            if tid not in self._tracker_bars:
                f = tk.Frame(self.tracker_frame, bg=BG2)
                f.pack(fill="x", pady=1)
                names = {1: "Hip", 2: "L.Foot", 3: "R.Foot", 4: "L.Knee", 5: "R.Knee"}
                lbl = tk.Label(f, text=f"{names.get(tid, f'T{tid}'):8s}", font=FONT_SM,
                               bg=BG2, fg=TEXT, width=8, anchor="w")
                lbl.pack(side="left")
                bar = ConfBar(f)
                bar.pack(side="left")
                self._tracker_bars[tid] = (lbl, bar)
            conf = t.get("confidence", 0)
            color = GREEN if conf > 0.6 else YELLOW if conf > 0.3 else RED
            self._tracker_bars[tid][0].config(fg=color)
            self._tracker_bars[tid][1].set(conf)

    def _create_feed_panels(self, num_cameras: int):
        """Create camera feed panels for the given number of cameras."""
        # Remove placeholder
        if self.feed_placeholder:
            self.feed_placeholder.pack_forget()
            self.feed_placeholder = None

        # Clear existing panels
        for panel in self._feed_panels.values():
            panel.destroy()
        self._feed_panels.clear()

        for i in range(num_cameras):
            panel = CameraFeedPanel(self.feed_container, cam_id=i)
            panel.pack(fill="x", pady=2, padx=2)
            self._feed_panels[i] = panel

    def _remove_feed_panels(self):
        """Remove feed panels and show placeholder."""
        for panel in self._feed_panels.values():
            panel.destroy()
        self._feed_panels.clear()
        self._preview_frames.clear()

        self.feed_placeholder = tk.Label(
            self.feed_container,
            text="Start server to see camera feeds",
            font=FONT, bg=BG3, fg=SUBTEXT, pady=80
        )
        self.feed_placeholder.pack(fill="both", expand=True)

    def _poll_preview(self):
        """Update camera feed previews at ~15fps."""
        if self._running:
            with self._preview_lock:
                for cam_id, data in self._preview_frames.items():
                    if cam_id in self._feed_panels:
                        self._feed_panels[cam_id].update_feed(
                            rgb_frame=data.get("rgb"),
                            landmarks=data.get("landmarks"),
                            bodies=data.get("bodies"),
                            fps=data.get("fps", 0.0),
                        )
        self.root.after(int(1000 / PREVIEW_FPS), self._poll_preview)

    def _toggle_server(self):
        if self._running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        args = self._collect_args()
        if args is None:
            return

        self._server_stop_event.clear()
        self._running = True
        self.btn_start.config(text="⏹  Stop Server", bg=RED, activebackground="#b91c1c")
        self.dot_server.set(YELLOW)
        self.lbl_server_state.config(text="Starting…", fg=YELLOW)
        self.lbl_statusbar.config(text="Starting server…")

        self._server_thread = threading.Thread(
            target=self._run_server, args=(args,), daemon=True, name="fbt-server"
        )
        self._server_thread.start()

    def _stop_server(self):
        self._server_stop_event.set()
        self._running = False
        self.btn_start.config(text="▶  Start Server", bg=GREEN, activebackground="#16a34a")
        self.dot_server.set(RED)
        self.lbl_server_state.config(text="Stopped", fg=RED)
        self.lbl_statusbar.config(text="Server stopped")
        self.root.after(1000, self._remove_feed_panels)

    def _run_server(self, args):
        """Run the server in background thread — mirrors server.py main()."""
        try:
            import camera as cam_module
            from platform_backend import create_backend, count_devices
            from calibration import load_calibration, build_default_calibration
            from fusion import MultiCameraFusion
            from osc_output import OSCOutput
            from debug_http import init_debug_server, start_debug_server
            cam_module.PLATFORM_BACKEND_FACTORY = create_backend

            num_cams = args["num_cameras"] or count_devices() or 1
            logging.info(f"Starting with {num_cams} camera(s) → {args['target_ip']}:{args['target_port']}")

            # Create feed panels in main thread
            self.root.after(0, lambda: self._create_feed_panels(num_cams))

            cameras = [cam_module.KinectCamera(i, fps_target=args["fps"]) for i in range(num_cams)]
            for c in cameras:
                c.start()

            self.root.after(0, lambda: (
                self.dot_server.set(GREEN),
                self.lbl_server_state.config(text="Running", fg=GREEN),
                self.lbl_statusbar.config(text=f"Running → {args['target_ip']}:{args['target_port']}"),
                self.lbl_osc.config(text=f"{args['target_ip']}:{args['target_port']}"),
            ))

            cal = load_calibration(args["cal_file"]) or build_default_calibration(num_cams)
            fusion = MultiCameraFusion(cal, user_height=args["user_height"])
            osc = OSCOutput(args["target_ip"], args["target_port"], dry_run=args["dry_run"])

            shared_state = {
                "cameras": cameras, "cameras_active": num_cams,
                "joints_tracked": 0, "fusion_fps": 0.0,
                "osc_target": f"{args['target_ip']}:{args['target_port']}",
                "joints": {}, "calibration": cal, "trackers": {},
                "preview_frames": self._preview_frames,  # shared ref — debug_http reads this
            }

            if args["debug"]:
                init_debug_server(shared_state)
                start_debug_server(8090)

            fps_counter = 0
            fps_time = time.monotonic()
            preview_time = time.monotonic()
            preview_interval = 1.0 / PREVIEW_FPS

            while not self._server_stop_event.is_set():
                # Read current frame interval (can change via slider)
                with self._frame_interval_lock:
                    frame_interval = self._current_frame_interval

                t0 = time.monotonic()
                frames = [f for c in cameras
                          for f in [c.get_frame(timeout=frame_interval * 0.4)]
                          if f is not None]

                if frames:
                    joints = fusion.update(frames)
                    trackers = fusion.get_trackers()
                    osc.send(trackers, len(frames), fusion.joints_tracked_count(), shared_state["fusion_fps"])

                    shared_state["joints"] = joints
                    shared_state["joints_tracked"] = fusion.joints_tracked_count()
                    shared_state["cameras_active"] = len(frames)
                    shared_state["trackers"] = {t.tracker_id: {"confidence": t.confidence,
                                                                "position": t.position}
                                                 for t in trackers}

                    # Update preview frames (throttled to preview FPS)
                    now_preview = time.monotonic()
                    if now_preview - preview_time >= preview_interval:
                        preview_time = now_preview
                        with self._preview_lock:
                            for frame in frames:
                                cam_id = frame.camera_id
                                # Detect multiple bodies
                                bodies = []
                                if frame.rgb_preview is not None:
                                    try:
                                        bodies = self._person_detector.detect(frame.rgb_preview)
                                    except Exception:
                                        pass

                                cam_obj = next((c for c in cameras if c.device_index == cam_id), None)
                                self._preview_frames[cam_id] = {
                                    "rgb": frame.rgb_preview,
                                    "landmarks": frame.landmarks,
                                    "bodies": bodies,
                                    "fps": cam_obj.fps if cam_obj is not None else 0.0,
                                }

                fps_counter += 1
                now = time.monotonic()
                if now - fps_time >= 1.0:
                    actual_fps = fps_counter / (now - fps_time)
                    shared_state["fusion_fps"] = actual_fps
                    fps_counter = 0
                    fps_time = now
                    self._push_status_update(shared_state)

                elapsed = time.monotonic() - t0
                sleep = frame_interval - elapsed
                if sleep > 0:
                    self._server_stop_event.wait(timeout=sleep)

            for c in cameras:
                c.stop()
            logging.info("Server stopped cleanly.")

        except Exception as e:
            logging.error(f"Server crashed: {e}", exc_info=True)
            self.root.after(0, lambda: (
                self.dot_server.set(RED),
                self.lbl_server_state.config(text="Error", fg=RED),
                messagebox.showerror("Server Error", str(e)),
            ))
            self._running = False
            self.root.after(0, lambda: self.btn_start.config(
                text="▶  Start Server", bg=GREEN, activebackground="#16a34a"))

    def _push_status_update(self, state: dict):
        target_fps = self._target_fps.get()
        actual_fps = state.get('fusion_fps', 0)

        def update():
            self.lbl_cameras.config(text=str(state.get("cameras_active", "—")))
            self.lbl_joints.config(text=str(state.get("joints_tracked", "—")))
            self.lbl_fps_actual.config(text=f"{actual_fps:.1f}")
            self.lbl_fps_target.config(text=f"/ {target_fps} target")
            # Color actual FPS based on how close to target
            ratio = actual_fps / target_fps if target_fps > 0 else 0
            if ratio >= 0.9:
                self.lbl_fps_actual.config(fg=GREEN)
            elif ratio >= 0.7:
                self.lbl_fps_actual.config(fg=YELLOW)
            else:
                self.lbl_fps_actual.config(fg=RED)
            self._update_tracker_ui(state.get("trackers", {}))
        self.root.after(0, update)

    def _run_calibration(self):
        if self._running:
            messagebox.showwarning("Server Running", "Stop the server before calibrating.")
            return
        args = self._collect_args()
        if args is None:
            return
        CalibrationWizard(self.root, args)

    def _collect_args(self) -> dict:
        try:
            ip = self.e_target_ip.get().strip()
            port = int(self.e_target_port.get().strip())
            fps = self._target_fps.get()
            height = float(self.e_height.get().strip())
            cal_file = self.e_cal_file.get().strip()
            num_cams_str = self.var_cams.get()
            num_cams = None if num_cams_str == "auto" else int(num_cams_str)
            return {
                "target_ip": ip, "target_port": port, "fps": fps,
                "user_height": height, "cal_file": cal_file,
                "num_cameras": num_cams,
                "dry_run": self.var_dryrun.get(),
                "debug": self.var_debug.get(),
            }
        except ValueError as e:
            messagebox.showerror("Invalid Settings", str(e))
            return None

    def _poll(self):
        """Drain log queue into the text widget every 100ms."""
        try:
            for _ in range(50):
                msg = self._log_queue.get_nowait()
                self.log_box.config(state="normal")
                tag = "ERROR" if "ERROR" in msg else "WARNING" if "WARNING" in msg else "INFO"
                self.log_box.insert("end", msg + "\n", tag)
                self.log_box.see("end")
                self.log_box.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _on_close(self):
        if self._running:
            self._stop_server()
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


# ── Calibration Wizard ────────────────────────────────────────────────────────
CHECKERBOARD = (9, 6)

class CalibrationWizard:
    """Modal calibration wizard with live camera feeds and checkerboard detection."""

    def __init__(self, parent, args: dict):
        self._args = args
        self._cameras = []
        self._running = False
        self._stop_event = threading.Event()
        self._checkerboard_status: Dict[int, bool] = {}  # cam_id -> detected
        self._capture_counts: Dict[str, int] = {}  # "a-b" -> frames captured
        self._current_pair_idx = 0
        self._capturing = False
        self._cal_thread = None

        self.win = tk.Toplevel(parent)
        self.win.title("Calibration Wizard")
        self.win.configure(bg=BG)
        self.win.geometry("1050x700")
        self.win.resizable(True, True)
        self.win.grab_set()

        self._build_ui()
        self._start_cameras()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.win, bg=ACCENT, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Calibration Wizard", font=(FONT[0], 14, "bold"),
                 bg=ACCENT, fg=TEXT).pack(side="left", padx=12)
        self.lbl_step = tk.Label(hdr, text="Starting cameras...", font=FONT,
                                 bg=ACCENT, fg="#d4b8ff")
        self.lbl_step.pack(side="left", padx=8)

        # Main area: camera feeds
        self.feeds_frame = tk.Frame(self.win, bg=BG)
        self.feeds_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # Status / instructions panel
        bot = tk.Frame(self.win, bg=BG2, pady=8)
        bot.pack(fill="x", padx=8, pady=(0, 4))

        self.lbl_instructions = tk.Label(
            bot, text="Waiting for cameras to start...",
            font=FONT, bg=BG2, fg=TEXT, wraplength=900, justify="left",
        )
        self.lbl_instructions.pack(fill="x", padx=8)

        # Per-camera checkerboard status
        self.status_frame = tk.Frame(bot, bg=BG2)
        self.status_frame.pack(fill="x", padx=8, pady=(4, 0))
        self._cam_status_labels: Dict[int, tk.Label] = {}

        # Progress bar for capture
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(bot, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(pady=(4, 0))

        # Buttons
        btn_frame = tk.Frame(self.win, bg=BG)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))

        self.btn_capture = tk.Button(
            btn_frame, text="Capture Pair", font=FONT_B,
            bg=GREEN, fg="white", relief="flat", pady=6, cursor="hand2",
            command=self._start_capture, state="disabled",
        )
        self.btn_capture.pack(side="left", padx=(0, 8))

        self.btn_skip = tk.Button(
            btn_frame, text="Skip Pair", font=FONT,
            bg=YELLOW, fg="black", relief="flat", pady=6, cursor="hand2",
            command=self._skip_pair, state="disabled",
        )
        self.btn_skip.pack(side="left", padx=(0, 8))

        self.btn_close = tk.Button(
            btn_frame, text="Cancel", font=FONT,
            bg=RED, fg="white", relief="flat", pady=6, cursor="hand2",
            command=self._close,
        )
        self.btn_close.pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self._close)

    def _start_cameras(self):
        def do_start():
            try:
                from platform_backend import count_devices, create_backend
                import camera as cam_module
                cam_module.PLATFORM_BACKEND_FACTORY = create_backend

                num_cams = self._args["num_cameras"] or count_devices() or 1
                self._cameras = [cam_module.KinectCamera(i, 15) for i in range(num_cams)]
                for c in self._cameras:
                    c.start()
                time.sleep(1.5)  # let cameras warm up

                self._running = True
                self.win.after(0, self._on_cameras_ready)
                self._poll_feeds()
            except Exception as e:
                logging.error(f"Calibration: camera start failed: {e}")
                err = str(e)
                self.win.after(0, lambda: messagebox.showerror(
                    "Camera Error", err, parent=self.win))
                self.win.after(0, self._close)

        threading.Thread(target=do_start, daemon=True).start()

    def _on_cameras_ready(self):
        num = len(self._cameras)

        # Create feed panels
        for widget in self.feeds_frame.winfo_children():
            widget.destroy()
        self._feed_canvases: Dict[int, tk.Canvas] = {}
        self._feed_photos: Dict[int, object] = {}

        for i, cam in enumerate(self._cameras):
            frame = tk.LabelFrame(self.feeds_frame, text=f"Camera {cam.device_index}",
                                  bg=BG2, fg=TEXT, font=FONT_B, relief="flat", padx=4, pady=4)
            frame.pack(side="left", fill="both", expand=True, padx=4, pady=4)

            canvas = tk.Canvas(frame, width=PREVIEW_W, height=PREVIEW_H,
                               bg="#0a0a14", highlightthickness=0)
            canvas.pack()
            self._feed_canvases[cam.device_index] = canvas

            # Status label per camera
            lbl = tk.Label(self.status_frame,
                           text=f"Cam {cam.device_index}: No checkerboard",
                           font=FONT, bg=BG2, fg=RED)
            lbl.pack(side="left", padx=(0, 16))
            self._cam_status_labels[cam.device_index] = lbl

        if num < 2:
            self.lbl_step.config(text="Single camera — saving identity calibration")
            self.lbl_instructions.config(
                text="Only 1 camera detected. Saving identity calibration (no stereo pairs needed)."
            )
            self.btn_capture.config(state="disabled")
            self.btn_skip.config(state="disabled")
            # Save single-camera calibration
            self._save_single_camera_cal()
        else:
            self._total_pairs = num - 1
            self._current_pair_idx = 0
            self._show_pair_instructions()

    def _save_single_camera_cal(self):
        from calibration import save_calibration
        import numpy as np
        cal = {self._cameras[0].device_index: np.eye(4)}
        save_calibration(cal, self._args["cal_file"])
        self.lbl_instructions.config(
            text=f"Calibration saved to {self._args['cal_file']}. You can close this window."
        )
        self.btn_close.config(text="Done", bg=GREEN)

    def _show_pair_instructions(self):
        idx = self._current_pair_idx
        cam_a = self._cameras[idx]
        cam_b = self._cameras[idx + 1]
        self.lbl_step.config(
            text=f"Pair {idx + 1}/{self._total_pairs}: "
                 f"Camera {cam_a.device_index} + Camera {cam_b.device_index}"
        )
        self.lbl_instructions.config(
            text=f"Hold the checkerboard ({CHECKERBOARD[0]}x{CHECKERBOARD[1]}) where BOTH "
                 f"Camera {cam_a.device_index} and Camera {cam_b.device_index} can see it.\n"
                 f"The status below will turn green when a camera detects the grid. "
                 f"Once both are green, click 'Capture Pair'."
        )
        self.btn_capture.config(state="normal")
        self.btn_skip.config(state="normal")
        self.progress_var.set(0)

    def _poll_feeds(self):
        """Background thread: grab frames, detect checkerboard, push to UI."""
        while self._running and not self._stop_event.is_set():
            for cam in self._cameras:
                frame = cam.get_frame(timeout=0.05)
                if frame is None or frame.rgb_preview is None:
                    continue

                rgb = frame.rgb_preview
                cam_id = cam.device_index

                # Detect checkerboard
                small = cv2.resize(rgb, (640, 360))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
                )
                self._checkerboard_status[cam_id] = found

                # Build preview image
                preview = cv2.resize(rgb, (PREVIEW_W, PREVIEW_H))
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

                if found:
                    # Scale corners to preview size
                    sx = PREVIEW_W / 640
                    sy = PREVIEW_H / 360
                    scaled_corners = corners.copy()
                    scaled_corners[:, :, 0] *= sx
                    scaled_corners[:, :, 1] *= sy
                    cv2.drawChessboardCorners(preview, CHECKERBOARD, scaled_corners, found)
                    # Green border
                    cv2.rectangle(preview, (2, 2), (PREVIEW_W - 3, PREVIEW_H - 3),
                                  (0, 220, 0), 3)
                else:
                    # Red border
                    cv2.rectangle(preview, (2, 2), (PREVIEW_W - 3, PREVIEW_H - 3),
                                  (200, 0, 0), 2)

                # Push to UI
                if Image is not None and ImageTk is not None:
                    pil_img = Image.fromarray(preview)
                    self.win.after(0, lambda cid=cam_id, img=pil_img: self._update_canvas(cid, img))

                # Update status label
                self.win.after(0, lambda cid=cam_id, f=found: self._update_status_label(cid, f))

            time.sleep(0.05)

    def _update_canvas(self, cam_id: int, pil_img):
        if cam_id not in self._feed_canvases:
            return
        try:
            photo = ImageTk.PhotoImage(pil_img)
            canvas = self._feed_canvases[cam_id]
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            self._feed_photos[cam_id] = photo  # prevent GC
        except Exception:
            pass

    def _update_status_label(self, cam_id: int, found: bool):
        if cam_id not in self._cam_status_labels:
            return
        if found:
            self._cam_status_labels[cam_id].config(
                text=f"Cam {cam_id}: Checkerboard DETECTED", fg=GREEN)
        else:
            self._cam_status_labels[cam_id].config(
                text=f"Cam {cam_id}: No checkerboard", fg=RED)

    def _start_capture(self):
        if self._capturing:
            return
        idx = self._current_pair_idx
        cam_a = self._cameras[idx]
        cam_b = self._cameras[idx + 1]

        # Check both cameras see the checkerboard
        a_ok = self._checkerboard_status.get(cam_a.device_index, False)
        b_ok = self._checkerboard_status.get(cam_b.device_index, False)
        if not a_ok or not b_ok:
            missing = []
            if not a_ok:
                missing.append(f"Camera {cam_a.device_index}")
            if not b_ok:
                missing.append(f"Camera {cam_b.device_index}")
            if not messagebox.askyesno(
                "Checkerboard Not Detected",
                f"{', '.join(missing)} cannot see the checkerboard right now.\n"
                "Capture anyway? (Will wait for both to detect it)",
                parent=self.win,
            ):
                return

        self._capturing = True
        self.btn_capture.config(state="disabled", text="Capturing...")
        self.btn_skip.config(state="disabled")

        def do_capture():
            try:
                self._capture_pair(cam_a, cam_b)
            except Exception as e:
                logging.error(f"Capture failed: {e}")
                self.win.after(0, lambda: messagebox.showerror(
                    "Capture Error", str(e), parent=self.win))
            finally:
                self._capturing = False
                self.win.after(0, self._on_pair_done)

        threading.Thread(target=do_capture, daemon=True).start()

    def _capture_pair(self, cam_a, cam_b):
        """Capture checkerboard frames from both cameras simultaneously."""
        from calibration import (CHECKERBOARD, SQUARE_SIZE_MM, NUM_FRAMES,
                                 _get_camera_intrinsics_matrix, save_calibration)
        import numpy as np

        objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        objp *= SQUARE_SIZE_MM

        obj_pts = []
        img_pts_a = []
        img_pts_b = []
        target = NUM_FRAMES
        timeout = time.monotonic() + 120

        while len(obj_pts) < target and time.monotonic() < timeout and not self._stop_event.is_set():
            fa = cam_a.get_frame(timeout=0.3)
            fb = cam_b.get_frame(timeout=0.3)
            if fa is None or fb is None or fa.rgb_preview is None or fb.rgb_preview is None:
                time.sleep(0.02)
                continue

            gray_a = cv2.cvtColor(cv2.resize(fa.rgb_preview, (960, 540)), cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(cv2.resize(fb.rgb_preview, (960, 540)), cv2.COLOR_BGR2GRAY)

            ret_a, corners_a = cv2.findChessboardCorners(gray_a, CHECKERBOARD, None)
            ret_b, corners_b = cv2.findChessboardCorners(gray_b, CHECKERBOARD, None)

            if ret_a and ret_b:
                obj_pts.append(objp)
                img_pts_a.append(corners_a)
                img_pts_b.append(corners_b)
                pct = len(obj_pts) / target * 100
                self.win.after(0, lambda p=pct: self.progress_var.set(p))
                logging.info(f"Calibration: captured frame {len(obj_pts)}/{target} "
                             f"(cam {cam_a.device_index}+{cam_b.device_index})")

            time.sleep(0.05)

        pair_key = f"{cam_a.device_index}-{cam_b.device_index}"
        self._capture_counts[pair_key] = len(obj_pts)

        if len(obj_pts) < 5:
            logging.warning(f"Only {len(obj_pts)} frames captured for pair {pair_key}")
            # Store empty result — will use identity
            self._pair_results = getattr(self, '_pair_results', {})
            self._pair_results[pair_key] = None
            return

        img_size = (960, 540)
        K_a = _get_camera_intrinsics_matrix(cam_a, img_size)
        K_b = _get_camera_intrinsics_matrix(cam_b, img_size)
        dist = np.zeros(5)

        try:
            ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
                obj_pts, img_pts_a, img_pts_b,
                K_a.copy(), dist.copy(),
                K_b.copy(), dist.copy(),
                img_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            mat_b_to_a = np.eye(4)
            mat_b_to_a[:3, :3] = R.T
            mat_b_to_a[:3, 3] = (-R.T @ T).ravel()

            self._pair_results = getattr(self, '_pair_results', {})
            self._pair_results[pair_key] = {
                "mat_b_to_a": mat_b_to_a,
                "rms": ret,
                "cam_a": cam_a.device_index,
                "cam_b": cam_b.device_index,
            }
            logging.info(f"Calibration pair {pair_key}: RMS={ret:.3f}")
        except Exception as e:
            logging.error(f"Stereo calibration failed for {pair_key}: {e}")
            self._pair_results = getattr(self, '_pair_results', {})
            self._pair_results[pair_key] = None

    def _skip_pair(self):
        idx = self._current_pair_idx
        cam_a = self._cameras[idx]
        cam_b = self._cameras[idx + 1]
        pair_key = f"{cam_a.device_index}-{cam_b.device_index}"
        self._pair_results = getattr(self, '_pair_results', {})
        self._pair_results[pair_key] = None
        logging.warning(f"Skipped calibration pair {pair_key} — will use identity")
        self._on_pair_done()

    def _on_pair_done(self):
        self._current_pair_idx += 1
        if self._current_pair_idx < self._total_pairs:
            self.btn_capture.config(state="normal", text="Capture Pair")
            self.btn_skip.config(state="normal")
            self._show_pair_instructions()
        else:
            self._finalize_calibration()

    def _finalize_calibration(self):
        """Build chained calibration from pair results and save."""
        import numpy as np
        from calibration import save_calibration

        calibration = {self._cameras[0].device_index: np.eye(4)}
        pair_results = getattr(self, '_pair_results', {})

        for i in range(len(self._cameras) - 1):
            cam_a = self._cameras[i]
            cam_b = self._cameras[i + 1]
            pair_key = f"{cam_a.device_index}-{cam_b.device_index}"
            result = pair_results.get(pair_key)

            if result is not None:
                cam_a_to_world = calibration[cam_a.device_index]
                calibration[cam_b.device_index] = cam_a_to_world @ result["mat_b_to_a"]
                logging.info(f"Camera {cam_b.device_index}: calibrated (RMS={result['rms']:.3f})")
            else:
                calibration[cam_b.device_index] = calibration.get(
                    cam_a.device_index, np.eye(4)).copy()
                logging.warning(f"Camera {cam_b.device_index}: using identity (no calibration data)")

        save_calibration(calibration, self._args["cal_file"])

        self.lbl_step.config(text="Calibration Complete")
        self.lbl_instructions.config(
            text=f"Calibration saved to {self._args['cal_file']}.\n"
                 f"Calibrated {len(calibration)} cameras. You can close this window."
        )
        self.btn_capture.pack_forget()
        self.btn_skip.pack_forget()
        self.btn_close.config(text="Done", bg=GREEN)

    def _close(self):
        self._running = False
        self._stop_event.set()
        for cam in self._cameras:
            try:
                cam.stop()
            except Exception:
                pass
        self.win.destroy()


def main():
    app = FBTServerGUI()
    app.run()


if __name__ == "__main__":
    main()
