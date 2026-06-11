# Using Marionette

Build first: [BUILD.md](BUILD.md). New to the project: [TUTORIAL.md](TUTORIAL.md).
Sensor placement and all calibration steps: [CALIBRATION.md](CALIBRATION.md).
Config schema reference: [DESIGN.md](DESIGN.md#config-schema-configjson).

The CLI is one binary with subcommands (`build/app/marionette`, or
`build\app\Release\marionette.exe` on Windows):

```
marionette list-types
marionette run -c <config.json> [--duration <sec>] [--verbose]
               [--no-dashboard] [--dashboard-port <p>]
marionette record -c <config.json> --node <id> -o <out.jsonl> [--duration <sec>]
marionette calibrate pair -c <config.json> --reference <id> --target <id>
                          [--min-samples <n>] [--max-seconds <sec>]
marionette calibrate body -c <config.json> [--seconds <sec>]
marionette calibrate playspace -c <config.json> [--hand left|right] [--seconds <sec>]
marionette doctor [-c <config.json>] [--json]
```

Exit codes: 0 success, 1 runtime failure, 2 usage error. Every subcommand
except `list-types` takes `-c <config.json>` (optional for `doctor`). Results
of the calibrate commands are written to the `calibration_file` named in the
config (default `calibration.json`).

## Zero-hardware demo

No Kinects, no headset, no SteamVR:

```sh
build/app/marionette run -c config/demo.json
```

`config/demo.json` runs two synthetic `mock` sensors (front + back, walking in
place with simulated depth noise) through the real fusion pipeline and sends
3-point OSC tracker output to `127.0.0.1:9001`, where nothing needs to be
listening.

While running, the app prints a periodic status line built from the pipeline
counters:

| Field | Meaning | Healthy demo value |
|---|---|---|
| `frames in` | Frames received from all capture nodes (cumulative) | grows ~60/s (2 nodes x 30 Hz) |
| `ticks` | Tick-loop iterations | grows at `tick_hz` (60/s) |
| `fused` | Ticks that produced a fused body | tracks `ticks` 1:1 |
| `trackers out` | Tracker poses pushed to endpoints (cumulative) | `fused` x 4 (3 trackers + head) |

`fused` falling behind `ticks` means no node had a fresh body that tick:
nobody in view, frames older than `fusion.stale_seconds`, or a stalled node.
The rig degrades gracefully - a dead node just stops contributing, and the
watchdog (see [Pipeline config keys](#pipeline-config-keys-phase-2)) restarts
nodes that go silent.

## Dashboard

`marionette run` serves a local web dashboard (when the build includes it,
the default; see [BUILD.md](BUILD.md#cmake-options)) at
**http://127.0.0.1:8211**. The HTTP API behind it is documented in
[DASHBOARD_API.md](DASHBOARD_API.md).

Tabs:

| Tab | What it shows |
|---|---|
| **Overview** | Pipeline counters (frames in, fused %, trackers out), per-node cards (fps, last-frame age, body, watchdog restarts, last error), endpoint list, calibration summary |
| **Skeleton** | Live fused body (front + top-down), sensor positions/view directions, tracker diamonds; polls ~15 Hz |
| **Calibration** | Guided wizards for pair / body / playspace with live progress and RMSE; one job at a time |
| **Events** | Structured warning/error feed from the in-process event log, filterable by level |
| **Tutorial** | Interactive 7-step onboarding (standalone copy: [TUTORIAL.md](TUTORIAL.md)); opens automatically on first visit |

Control it per run:

- `--no-dashboard` disables it for this run.
- `--dashboard-port <p>` overrides the configured port.
- Persistent settings live in the config's `dashboard` object
  (`enable`, `bind`, `port` - see
  [Pipeline config keys](#pipeline-config-keys-phase-2)).

The dashboard binds loopback (`127.0.0.1`) by default and has no auth. To view
it from another machine on your LAN or tailnet, set `"bind": "0.0.0.0"` in the
config's `dashboard` object - an explicit choice, since anyone who can reach
the port can run calibration jobs.

## doctor

`marionette doctor [-c <config.json>] [--json]` is the pre-flight check: it
probes compiled backends and SDK environment variables, starts each attached
sensor briefly (~3 s per sensor) and counts frames, validates the config (node
and endpoint types against this build), summarizes the calibration file,
checks the wire UDP port, and looks for the SteamVR driver registration.

```
marionette doctor -c config.json

probing sensors (~3 s per sensor)...

marionette doctor
  version              0.1.0
  platform             windows

backends
  kinect_v2            compiled
  kinect_v1            not compiled
  openvr_client        compiled
  dashboard            compiled
  KINECTSDK20_DIR      C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\
  KINECTSDK10_DIR      (not set)

sensors
  kinect_v2            OK: 89 frames in 3.0 s (29.7 fps)

config
  file                 config.json
  load                 OK: 2 node(s), 1 endpoint(s)

calibration
  file                 calibration.json
  extrinsics           2 (front, back)
  body model           valid
  world anchor         not set

ports
  udp 24190 (wire)     free
  dashboard            8211 (configured; not probed)

steamvr
  openvrpaths          found: C:\Users\you\AppData\Local\openvr\openvrpaths.vrpath
  driver registered    yes

result: OK
```

Exit codes: **0** = no problems, **1** = problems found (each listed under
`result:`), **2** = usage error. A sensor that is merely absent is reported
`not present`, not a problem; a sensor that starts but delivers no frames is.
`--json` prints a single machine-readable JSON document instead (same content
plus a `problems` array and an `ok` bool) and silences log lines below error.

## Pipeline config keys (phase 2)

Three config objects beyond the phase-1 schema in
[DESIGN.md](DESIGN.md#config-schema-configjson) - all optional, defaults
shown:

```jsonc
{
  "watchdog": {
    "enable": true,
    "silent_seconds": 5.0,     // no frame for this long -> restart the node
    "backoff_seconds": 5.0,    // min delay between restart attempts per node
    "max_restarts": 10,        // per node, per run
    "exclude_types": ["replay"] // node types that legitimately go quiet
  },
  "dashboard": {
    "enable": true,
    "bind": "127.0.0.1",       // "0.0.0.0" exposes on LAN/tailnet (no auth)
    "port": 8211
  },
  "fusion": {
    // ...phase-1 keys..., plus cross-node outlier rejection:
    "outlier_rejection": true,
    "outlier_threshold_m": 0.35,   // distance from other nodes' consensus
    "outlier_weight_factor": 0.05  // weight multiplier for flagged samples
  }
}
```

The watchdog recreates and restarts a capture node that stops delivering
frames (see [DESIGN.md](DESIGN.md#watchdog)); restarts show up per node on the
dashboard Overview tab and in the events feed. Outlier rejection downweights a
sensor whose joint sample disagrees with the consensus of the other sensors
(see [DESIGN.md](DESIGN.md#outlier-rejection)) - the usual culprit is a
bumped, miscalibrated sensor.

## Quest standalone (VRChat OSC)

Tracking data goes straight to the headset over Wi-Fi. No PCVR stream, no
SteamVR, no driver - the PC only runs Marionette and the Kinects.

1. Get the headset's IP (Quest: Settings > Wi-Fi > your network > details).
   A static DHCP lease helps.
2. Point the `osc` endpoint at it in your config:

   ```json
   "endpoints": [ { "type": "osc", "params": { "host": "192.168.1.50", "port": 9000 } } ]
   ```

   9000 is VRChat's OSC input port. Keep `"emit_head": true` in `mapping`
   (the default): VRChat aligns and scales the incoming tracker space to your
   avatar using the head reference Marionette sends, so no playspace
   calibration is needed on this path.
3. In VRChat, enable OSC: Action Menu > Options > OSC > Enabled.
4. Run `marionette run -c <config.json>`, then in VRChat open the main menu
   and run FBT calibration (stand straight, arms out, confirm). Recalibrate
   in-game whenever trackers look offset.

Tracker count is the `mapping.trackers` list in the config (VRChat accepts up
to 8):

| Setup | `mapping.trackers` |
|---|---|
| 3-point (start here) | `["waist","left_foot","right_foot"]` |
| 6-point | + `"chest","left_knee","right_knee"` |
| 8-point | + `"left_elbow","right_elbow"` |

3-point gives full-body IK in VRChat (head + hands come from the headset).
Add knees/chest once 3-point is solid - they amplify any calibration error.
Elbows are only worth it with sensors that see your arms clearly.

## PCVR (SteamVR)

For VRChat or any OpenVR app via ALVR, Virtual Desktop, Steam Link, or Quest
Link (Quest 2/3), or any wired PCVR headset.

1. Build with the driver enabled (default) and register it once:
   `scripts/install-driver.ps1` / `.sh` (see
   [BUILD.md](BUILD.md#steamvr-driver-output-and-install)).
2. Add the `openvr` endpoint to your config:

   ```json
   "endpoints": [ { "type": "openvr", "params": { "host": "127.0.0.1", "port": 24190 } } ]
   ```

   Both endpoints can be active simultaneously.
3. Start SteamVR (through ALVR / Virtual Desktop / Link as usual), then
   `marionette run -c <config.json>`. The driver creates one generic tracker
   per role the first time it sees it (serials `MN-WAIST`, `MN-LFOOT`,
   `MN-RFOOT`, ...). Trackers go invalid if Marionette stops sending for
   0.5 s.
4. Calibrate the playspace anchor so Marionette's world lines up with the
   SteamVR playspace: `marionette calibrate playspace`
   ([CALIBRATION.md](CALIBRATION.md#4-playspace-anchor)). Until then the two
   spaces coincide only by luck.
5. Assign roles: SteamVR > Settings > Controllers > Manage Trackers, set
   `MN-WAIST` to Waist, `MN-LFOOT` to Left Foot, and so on. SteamVR remembers
   this per serial. Then calibrate FBT inside VRChat as usual.

**Quest 2**: identical logic to Quest 3, just budget the stream tighter -
lower the video bitrate in ALVR / Virtual Desktop. Tracker data is tiny; it is
the video stream that competes for bandwidth.

## Recording and replaying sessions

Record the raw frames of one capture node to JSONL:

```sh
marionette record -c config.json --node front -o session-front.jsonl
```

`--node <id>` names which config node to run (required); recording runs just
that node and writes its local-frame stream to the `-o` file. To capture a
multi-sensor session, run `record` once per node with different output files.
Stop with Ctrl+C, or pass `--duration <sec>`. The format is one frame per
line: `{"t": <seconds>, "has_body": true, "joints": [[x,y,z, qw,qx,qy,qz,
confidence, state], ...]}`.

Play a recording back through the full pipeline by swapping the node type to
`replay` in a config:

```json
"nodes": [ { "id": "front", "type": "replay",
             "params": { "file": "session-front.jsonl", "loop": true, "speed": 1.0,
                         "extrinsic": { "pos": [0, 0.9, -2.0], "rot": [1, 0, 0, 0] } } } ]
```

This is the tuning loop: record a session once (wearing the headset is not
required), then iterate `fusion` settings (One-Euro `filter` params,
`occlusion_penalty`, `bone_length_constraint`, ...) against the identical,
deterministic input until the output is right. `speed` slows playback for
frame-by-frame inspection; `loop` keeps it running.
