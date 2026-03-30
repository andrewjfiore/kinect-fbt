# CODEX — Kinect FBT Volumetric Streaming

## What was built (2026-03-30)

Volumetric 3D point cloud streaming + native Kinect SDK skeleton tracking for the
web-based monitoring UI. This is the verification/monitoring layer before Meta Quest
integration.

### New files
- `kinect_server/pointcloud.py` — RGB-D to colored 3D point cloud (depth-based coloring, camera intrinsic unprojection, binary WebSocket protocol)
- `kinect_server/web_server.py` — aiohttp WebSocket server replacing Flask for real-time binary streaming
- `kinect_server/web/index.html` — Three.js frontend: 3 viewports (cam0, fused, cam1), orbit controls, pre-allocated GPU buffers, cylinder bone skeleton, tracker joint highlights
- `kinect_server/platform_backend.py::WindowsKinectV1Backend` — ctypes COM wrapper for Kinect10.dll (v1 on Windows), including `NuiSkeletonGetNextFrame` for native skeleton
- `tests/test_volumetric.py` — 23 E2E tests for point cloud pipeline
- `run_volumetric.bat` — One-click launcher (auto-creates Python 3.10 venv, patches pykinect2)
- `patch_pykinect2.py` — Fixes pykinect2 struct assertion + comtypes version check for 64-bit Python

### Modified files
- `kinect_server/server.py` — `--web-server` flag, alternating 15Hz per-camera point cloud streaming, skeleton dict builder
- `kinect_server/camera.py` — Native Kinect SDK skeleton (priority 1) with MediaPipe fallback (priority 2), `KINECT_SDK_TO_JOINT_NAME` mapping, `rgb_clean` field
- `kinect_server/platform_backend.py` — v2 backend now requests Body frames, `get_body_joints()` on both v1 and v2 backends, Windows v1 enumeration
- `requirements*.txt` — Added `aiohttp>=3.9.0`
- `tests/conftest.py` — Added aiohttp mock

### How to run
```
run_volumetric.bat
# Then open http://localhost:8090
# Stand 5-6 feet from Kinects with full body visible
```

---

## Outstanding issues

### 1. SKELETON IS UPSIDE DOWN relative to point cloud
**Status**: NOT FIXED
**Symptom**: The green skeleton bones render inverted (as if hanging from ceiling) while the depth point cloud shows the room right-side up.
**Root cause**: Coordinate system mismatch between the point cloud unprojection and the Kinect SDK skeleton data.
- Point cloud: image pixel rows (Y increases downward) are unprojected and Y is negated → Y-up
- Kinect SDK skeleton: already Y-up natively (positive Y = above sensor)
- Both should be Y-up and aligned, but the fusion pipeline's origin correction (`fusion.py` `_try_set_origin`) applies an offset to skeleton joints that is NOT applied to the point cloud, creating a vertical displacement
- The calibration transforms may also be contributing — the default "facing" layout places cameras at z=+/-2m with rotation, which could flip Y depending on the rotation matrix

**Possible fixes**:
1. Apply the fusion origin offset to the point cloud as well (in `server.py` before encoding)
2. Skip the fusion origin correction for the web skeleton and send raw world-space positions for both
3. Debug by logging actual Y values of skeleton joints vs point cloud centroids and comparing

### 2. KINECT V2 BODY TRACKING not returning data
**Status**: PARTIALLY WORKING
**Symptom**: `get_body_joints()` on `WindowsKinect2Backend` returns `None` most frames even when a person is visible.
**Root cause**: The `has_new_body_frame()` check in `get_frames()` may miss body frames due to timing — body frames arrive at 30fps but `get_frames()` is called at the server's target FPS (20Hz). If the body frame check and color/depth check don't align, body data is lost.
**Possible fixes**:
1. Cache the body frame separately with its own polling, don't gate it on `has_new_body_frame()` in the same call as color/depth
2. Use `get_last_body_frame()` unconditionally (it returns the most recent, even if not "new")
3. Switch to event-driven body frame arrival like Amethyst does

### 3. KINECT V1 SKELETON returns E_NUI_FRAME_NO_DATA frequently
**Status**: INFRASTRUCTURE DONE, DATA SPARSE
**Symptom**: `NuiSkeletonGetNextFrame` returns `0x83010001` (no data) on many frames. When a person IS detected, the 20-joint skeleton works.
**Root cause**: The v1 skeleton tracker requires the full body silhouette to be clearly visible in the depth stream. It's more strict than v2 about body visibility. Also, the `NuiSkeletonGetNextFrame` timeout is set to 0ms (non-blocking), which may miss frames.
**Possible fixes**:
1. Increase timeout to 33ms (one frame period)
2. Ensure `DEPTH_AND_PLAYER_INDEX` stream is being consumed (currently it is)
3. Add `NuiTransformSmooth` call (vtable[20]) to smooth skeleton data

### 4. POINT CLOUD uses depth-based coloring, not RGB
**Status**: BY DESIGN
**Explanation**: RGB-to-depth registration on Kinect requires per-pixel coordinate mapping. The depth and color sensors have different FOVs and positions. Without proper registration, RGB colors map to wrong 3D positions. Depth-based coloring (cyan=near, orange=far) avoids this entirely and gives accurate spatial representation.
**To add RGB mode**: Pass `color_mode="rgb"` to `rgbd_to_pointcloud()`. It works but colors will be misaligned unless the depth map is properly registered to color space (v2's `_map_depth_to_color` does this, v1 does not).

### 5. PYKINECT2 requires Python 3.10
**Status**: WORKAROUND IN PLACE
**Symptom**: pykinect2's ctypes struct assertions fail on Python 3.11+ (64-bit struct size 80 vs expected 72, comtypes version mismatch).
**Workaround**: `patch_pykinect2.py` patches the installed package. `run_volumetric.bat` auto-creates a Python 3.10 venv and applies the patch.
**Long-term fix**: Replace pykinect2 with direct ctypes wrapper against `Kinect20.dll` (like we did for v1 with `Kinect10.dll`).

### 6. CI/CD not updated for new modules
**Status**: NOT DONE
**What's needed**:
- `.github/workflows/build.yml`: add `aiohttp` to test deps, include `pointcloud.py`, `web_server.py`, `web/` in build artifacts
- `kinect_server/fbt_server.spec`: add hidden imports for `pointcloud`, `web_server`, `aiohttp`
- `packaging/windows/build_exe.py` and `packaging/linux/build_linux.sh`: add `--add-data` entries

### 7. ONLY 1 CAMERA contributes point clouds
**Status**: KNOWN
**Symptom**: Server logs show "1 cams" in point cloud generation even though both cameras are open.
**Root cause**: The v1 backend's depth frames may be all zeros or out of valid range, causing `rgbd_to_pointcloud` to produce 0 points. The v1 depth stream switched to `DEPTH_AND_PLAYER_INDEX` mode which changes the depth format — the depth extraction code handles this (`raw >> 3`) but needs verification that valid depth data is being produced.

## Architecture reference

```
Kinect v1 (360) ─→ Kinect10.dll (ctypes) ─→ RGB + Depth + 20-joint skeleton
Kinect v2 (One) ─→ pykinect2 (SDK v2)    ─→ RGB + Depth + 25-joint skeleton
                                               │
                              camera.py: native skeleton (priority) or MediaPipe (fallback)
                                               │
                              fusion.py: multi-cam weighted average + 1-euro filter
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              OSC → VRChat              pointcloud.py              web_server.py
              (tracker data)         (depth → 3D points)       (WebSocket binary)
                                           │                          │
                                     encode_binary ──────→ Three.js web UI
                                                          (3 viewports + skeleton)
```
