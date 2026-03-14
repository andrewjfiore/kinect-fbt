"""
kinect_server/gui.py — Cross-platform tkinter GUI for the FBT server.
Works on Windows and Linux. Bundles into a single executable via PyInstaller.

One-click Start/Stop with live status dashboard.
References: SlimeVR Server GUI, KinectToVR (K2EX) UI patterns.
"""
import platform
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging

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
ACCENT  = "#7c3aed"  # purple
GREEN   = "#22c55e"
YELLOW  = "#eab308"
RED     = "#ef4444"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
FONT    = ("Segoe UI" if IS_WINDOWS else "Ubuntu", 10)
FONT_B  = (FONT[0], 10, "bold")
FONT_H  = (FONT[0], 13, "bold")
FONT_SM = (FONT[0], 8)


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


# ── Main GUI ─────────────────────────────────────────────────────────────────
class FBTServerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FBT Server — Kinect v2 Full-Body Tracking")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(700, 560)

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

        self._build_ui()
        self._setup_logging()
        self._poll()

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

        # ── Main layout: left (controls) + right (log) ───────────────────
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(main, bg=BG, width=340)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(main, bg=BG)
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
        self.e_fps         = row("FPS",              "20")
        self.e_height      = row("User Height (m)",  "1.70")
        self.e_cal_file    = row("Calibration File", "calibration.json")

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
        self.lbl_fps       = row("Fusion FPS")
        self.lbl_osc       = row("OSC Target")

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

    def _toggle_server(self):
        if self._running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        # Build args from UI
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

    def _run_server(self, args):
        """Run the server in background thread — mirrors server.py main()."""
        try:
            import camera as cam_module
            from platform_backend import create_backend, count_devices
            from calibration import load_calibration, build_default_calibration
            from fusion import MultiCameraFusion
            from osc_output import OSCOutput
            from debug_http import init_debug_server, start_debug_server
            import importlib
            # Patch camera module to use platform backend
            cam_module.PLATFORM_BACKEND_FACTORY = create_backend

            num_cams = args["num_cameras"] or count_devices() or 1
            logging.info(f"Starting with {num_cams} camera(s) → {args['target_ip']}:{args['target_port']}")

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
            }

            if args["debug"]:
                init_debug_server(shared_state)
                start_debug_server(8090)

            frame_interval = 1.0 / args["fps"]
            fps_counter = 0
            fps_time = time.monotonic()

            while not self._server_stop_event.is_set():
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

                fps_counter += 1
                now = time.monotonic()
                if now - fps_time >= 1.0:
                    shared_state["fusion_fps"] = fps_counter / (now - fps_time)
                    fps_counter = 0
                    fps_time = now
                    # Update UI from main thread
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
        def update():
            self.lbl_cameras.config(text=str(state.get("cameras_active", "—")))
            self.lbl_joints.config(text=str(state.get("joints_tracked", "—")))
            self.lbl_fps.config(text=f"{state.get('fusion_fps', 0):.1f}")
            self._update_tracker_ui(state.get("trackers", {}))
        self.root.after(0, update)

    def _run_calibration(self):
        if self._running:
            messagebox.showwarning("Server Running", "Stop the server before calibrating.")
            return
        args = self._collect_args()
        if args is None:
            return

        def do_cal():
            try:
                from platform_backend import count_devices, create_backend
                import camera as cam_module
                cam_module.PLATFORM_BACKEND_FACTORY = create_backend
                from calibration import run_calibration
                num_cams = args["num_cameras"] or count_devices() or 1
                cameras = [cam_module.KinectCamera(i, 15) for i in range(num_cams)]
                for c in cameras:
                    c.start()
                time.sleep(2)
                logging.info("Starting calibration — place checkerboard now!")
                run_calibration(cameras, filepath=args["cal_file"])
                for c in cameras:
                    c.stop()
                self.root.after(0, lambda: messagebox.showinfo(
                    "Calibration", f"Done! Saved to {args['cal_file']}"))
            except Exception as e:
                logging.error(f"Calibration failed: {e}")
                self.root.after(0, lambda: messagebox.showerror("Calibration Error", str(e)))

        threading.Thread(target=do_cal, daemon=True).start()

    def _collect_args(self) -> dict:
        try:
            ip = self.e_target_ip.get().strip()
            port = int(self.e_target_port.get().strip())
            fps = int(self.e_fps.get().strip())
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


def main():
    app = FBTServerGUI()
    app.run()


if __name__ == "__main__":
    main()
