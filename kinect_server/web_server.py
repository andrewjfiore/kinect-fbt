"""
web_server.py — WebSocket + HTTP server for volumetric 3D streaming.
Replaces the Flask debug_http.py with a full-featured monitoring server.

Streams:
  - Per-camera point clouds (binary) at 15Hz each (alternating for 30Hz combined)
  - Fused point cloud (binary)
  - Skeleton joint data (embedded in point cloud frames)
  - Status JSON (periodic)

Serves the Three.js web frontend from kinect_server/web/
"""
import asyncio
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Try to import aiohttp; fall back gracefully
try:
    from aiohttp import web
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("aiohttp not installed — web server disabled. Install: pip install aiohttp")


class VolumetricWebServer:
    """
    Async WebSocket server that streams point cloud + skeleton data to browser clients.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8090):
        self.host = host
        self.port = port
        self._clients: Set = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._status: Dict[str, Any] = {
            "cameras_active": 0,
            "joints_tracked": 0,
            "fusion_fps": 0.0,
            "osc_target": "",
            "web_clients": 0,
        }
        # Static file directory
        self._web_dir = Path(__file__).parent / "web"

    def start(self):
        """Start the web server in a background thread."""
        if not HAS_AIOHTTP:
            logger.error("Cannot start web server: aiohttp not installed")
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True, name="web-server")
        self._thread.start()
        logger.info(f"Volumetric web server starting on http://{self.host}:{self.port}")

    def _run_server(self):
        """Run the asyncio event loop in a dedicated thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_app())
        self._loop.run_forever()

    async def _start_app(self):
        self._app = web.Application()
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/api/status", self._status_handler)
        self._app.router.add_get("/api/health", self._health_handler)

        # Serve static files from web/ directory
        if self._web_dir.exists():
            self._app.router.add_get("/", self._index_handler)
            self._app.router.add_static("/static/", path=str(self._web_dir), name="static")
        else:
            logger.warning(f"Web directory not found: {self._web_dir}")

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Volumetric web server running on http://{self.host}:{self.port}")

    async def _index_handler(self, request):
        index_path = self._web_dir / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Web UI not found", status=404)

    async def _health_handler(self, request):
        return web.json_response({"status": "ok", "timestamp": time.time()})

    async def _status_handler(self, request):
        with self._lock:
            status = self._status.copy()
        status["web_clients"] = len(self._clients)
        return web.json_response(status)

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)  # 16MB max
        await ws.prepare(request)

        self._clients.add(ws)
        client_ip = request.remote
        logger.info(f"WebSocket client connected: {client_ip} (total: {len(self._clients)})")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client commands
                    try:
                        cmd = json.loads(msg.data)
                        await self._handle_client_command(ws, cmd)
                    except json.JSONDecodeError:
                        pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
        finally:
            self._clients.discard(ws)
            logger.info(f"WebSocket client disconnected: {client_ip} (total: {len(self._clients)})")

        return ws

    async def _handle_client_command(self, ws, cmd: dict):
        """Handle commands from web clients (e.g., request status, change settings)."""
        action = cmd.get("action")
        if action == "get_status":
            with self._lock:
                status = self._status.copy()
            status["web_clients"] = len(self._clients)
            await ws.send_str(json.dumps({"type": "status", "data": status}))

    def broadcast_pointcloud(self, data: bytes):
        """
        Send a binary point cloud frame to all connected WebSocket clients.
        Called from the main server thread — schedules async sends on the event loop.
        """
        if not self._loop or not self._clients:
            return

        async def _send():
            dead = set()
            for ws in self._clients.copy():
                try:
                    await ws.send_bytes(data)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    def broadcast_status(self, status: dict):
        """Send a JSON status update to all connected clients."""
        if not self._loop or not self._clients:
            return

        msg = json.dumps({"type": "status", "data": status})

        async def _send():
            dead = set()
            for ws in self._clients.copy():
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    def update_status(self, **kwargs):
        """Update internal status dict (thread-safe)."""
        with self._lock:
            self._status.update(kwargs)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def stop(self):
        """Shutdown the server."""
        if self._loop:
            async def _shutdown():
                # Close all WebSocket connections
                for ws in self._clients.copy():
                    await ws.close()
                if self._runner:
                    await self._runner.cleanup()

            try:
                future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
                future.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
