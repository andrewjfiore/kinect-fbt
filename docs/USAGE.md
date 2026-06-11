# Using Marionette

Build first: [BUILD.md](BUILD.md). Sensor placement and all calibration steps:
[CALIBRATION.md](CALIBRATION.md). Config schema reference:
[DESIGN.md](DESIGN.md#config-schema-configjson).

The CLI is one binary with subcommands (`build/app/marionette`, or
`build\app\Release\marionette.exe` on Windows):

```
marionette run                 -c <config.json>     # capture -> fuse -> endpoints
marionette record              -c <config.json> -o <out.jsonl>
marionette calibrate pair      -c <config.json>     # sensor-to-sensor extrinsics
marionette calibrate body      -c <config.json>     # per-user bone lengths
marionette calibrate playspace -c <config.json>     # SteamVR anchor (needs SteamVR)
```

Every subcommand takes `-c <config.json>`. Results of the calibrate commands
are written to the `calibration_file` named in the config
(default `calibration.json`).

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
The rig degrades gracefully - a dead node just stops contributing.

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

Record the raw per-node frames of a live session to JSONL:

```sh
marionette record -c config.json -o session.jsonl
```

Recording runs the configured capture nodes and writes each node's local-frame
stream (one JSONL file per node; the node id is added to the file name when
there is more than one). Stop with Ctrl+C. The format is one frame per line:
`{"t": <seconds>, "has_body": true, "joints": [[x,y,z, qw,qx,qy,qz, confidence,
state], ...]}`.

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
