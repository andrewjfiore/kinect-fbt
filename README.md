# Marionette

Multi-sensor full-body tracking for SteamVR and Quest (VRChat OSC), built around a variable
number of Microsoft Kinect sensors (v1 and v2). Windows and Linux.

Strings from above, moving a body.

## What it does

- Captures skeletal data from N Kinect sensors placed around your play space
  (Kinect v2 and v1 via the Microsoft SDKs on Windows; markerless libfreenect2 backend planned).
- Calibrates the sensor network (sensor-to-sensor Kabsch/Horn rigid alignment) and anchors it
  to your SteamVR play space.
- Fuses all views into one skeleton: per-joint confidence/occlusion/depth-noise weighting,
  One-Euro filtering, bone-length constraints. Multiple viewpoints eliminate the classic
  single-Kinect failure modes (front/back flip ambiguity, self-occlusion dropouts).
- Outputs full-body trackers two ways, simultaneously if you want:
  - **SteamVR**: an OpenVR driver exposing virtual generic trackers (works with ALVR,
    Virtual Desktop, Steam Link, Quest Link for Quest 2/3 PCVR).
  - **VRChat OSC Trackers**: direct to the headset over LAN, no PCVR stream required
    (Quest standalone).
- Serves a local web dashboard at http://127.0.0.1:8211 while running: live skeleton view,
  pipeline status, event feed, calibration wizards, and an onboarding tutorial.
- `marionette doctor` health-checks the whole stack (backends, sensors, config,
  calibration, ports, SteamVR driver registration), with `--json` for scripts.
- Self-heals at runtime: a watchdog recreates and restarts capture nodes that go silent,
  and structured warning/error events are tracked in-process (dashboard Events tab).

## Layout

| Path | What |
|---|---|
| `core/` | Joint schema, capture/endpoint interfaces, fusion, calibration, filters, pipeline |
| `capture/mock/` | Synthetic + replay capture nodes (dev/test without hardware) |
| `capture/kinect_v2/` | Kinect v2 via Kinect SDK 2.0 (Windows, auto-enabled when SDK present) |
| `capture/kinect_v1/` | Kinect v1 (Xbox 360) via Kinect SDK 1.8 (Windows, auto-enabled) |
| `endpoints/osc/` | VRChat OSC Trackers endpoint |
| `endpoints/openvr_bridge/` | UDP bridge feeding the OpenVR driver |
| `driver/openvr/` | `driver_marionette` SteamVR driver (virtual trackers) |
| `server/` | Web dashboard: embedded UI + JSON API (cpp-httplib) |
| `web/` | The dashboard's single-file UI (`index.html`, embedded at build time) |
| `app/` | `marionette` CLI: run, record, calibrate, doctor |
| `docs/` | Design, build, usage, calibration, tutorial, dashboard API |

## Quick start

See [docs/BUILD.md](docs/BUILD.md) and [docs/USAGE.md](docs/USAGE.md); first time,
start with [docs/TUTORIAL.md](docs/TUTORIAL.md).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release
ctest --test-dir build -C Release -L hardware  # optional: smoke-test attached sensors
build/app/marionette run -c config/demo.json   # no hardware needed (mock nodes)
```

On Windows the Visual Studio generator is multi-config, so the binary lands at
`build\app\Release\marionette.exe` (not `build/app/`). The demo run also serves the
dashboard at http://127.0.0.1:8211.

## Standing on the shoulders of

Architecture informed by [Amethyst/K2VR](https://github.com/KinectToVR) (device/endpoint plugin
model, SteamVR + OSC dual output), [OpenVR-SpaceCalibrator](https://github.com/pushrax/OpenVR-SpaceCalibrator)
(playspace anchoring math), [SlimeVR](https://github.com/SlimeVR) (body proportions + OSC
conventions), libfreenect/libfreenect2, and the One-Euro filter (Casiez et al.).
Amethyst itself is Windows-only; Marionette is a standalone cross-platform core so the same
code runs on Windows and Linux.

License: MIT (see LICENSE).
