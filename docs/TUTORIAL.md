# Tutorial: from zero to trackers on your body

This is the standalone version of the dashboard's built-in tutorial (the
Tutorial tab). It assumes nothing: no prior setup, no calibration, no
familiarity with the project. Each step names the dashboard tab that does the
job and the CLI command that does the same thing without a browser.

Prerequisites: a built binary ([BUILD.md](BUILD.md)). Hardware is optional
until step 6 - steps 1-5 work fully against the zero-hardware demo config.

Quick orientation: `marionette run` serves a local web dashboard at
**http://127.0.0.1:8211** (when built with `MN_BUILD_DASHBOARD`, the default).
Its tabs are **Overview** (pipeline counters, node cards), **Skeleton** (live
fused body), **Calibration** (wizards for all three calibrations), **Events**
(warnings/errors), and **Tutorial** (this walkthrough, interactive). See
[USAGE.md](USAGE.md#dashboard) for the full tab reference.

## Step 1 - Welcome

Marionette turns one or more Kinect sensors into full-body tracking for VR.
Each sensor watches you from a different angle; the fusion engine merges every
view into one stable skeleton and streams tracker poses to SteamVR or straight
to a Quest headset over Wi-Fi. Opposed viewpoints structurally remove the two
classic single-Kinect failures: front/back flips and self-occlusion dropouts.

**Try it right now, no hardware:**

```sh
marionette run -c config/demo.json
```

This runs two synthetic sensors (front + back, walking in place with simulated
depth noise) through the real fusion pipeline and starts the dashboard. Open
http://127.0.0.1:8211 - everything in this tutorial can be dry-run against it.

- Dashboard: the Tutorial tab opens automatically on first visit.
- CLI: `marionette -h` lists every subcommand; `marionette list-types` shows
  which capture node and endpoint types are compiled into your binary.

## Step 2 - Choose your path

Two ways to get trackers into your game. Both endpoints can run at once, but
pick one to set up first.

| | Quest standalone (simplest) | PCVR / SteamVR (most compatible) |
|---|---|---|
| Transport | VRChat OSC over Wi-Fi, no PCVR stream | OpenVR driver via ALVR / Virtual Desktop / Link / wired |
| Needs SteamVR | No | Yes |
| Needs driver install | No | Yes, one time |
| Needs playspace calibration | No | Yes |
| Trackers | Up to 8 via OSC | Real SteamVR devices (`MN-*` serials) |

Full setup details for each path: [USAGE.md](USAGE.md#quest-standalone-vrchat-osc)
and [USAGE.md](USAGE.md#pcvr-steamvr).

**Pre-flight check** before touching hardware:

```sh
marionette doctor -c config.json
```

`doctor` probes compiled backends, attached sensors, your config, the
calibration file, ports, and SteamVR registration, then prints `result: OK` or
a problem list ([USAGE.md](USAGE.md#doctor)). The dashboard mirrors a
lightweight version of this on its Overview tab.

## Step 3 - Place your sensors

- Mount at roughly head height (1.6-2.0 m), tilted slightly down, play-area
  center 2-3.5 m away. Depth noise grows with range; past ~4 m a sensor
  contributes little.
- Two sensors cover most of 360 degrees: place them opposing (front + back) or
  at a 90-degree offset (front + side).
- One Kinect v2 per PC (hard SDK limit); multiple v1s can share a PC.
- Never overlap two Kinect v1s (their IR dot patterns interfere). v1 + v2 and
  v2 + v2 mixes are clean.
- Mount rigidly: a bumped tripod means redoing pair calibration for that
  sensor.

Full placement guidance: [CALIBRATION.md](CALIBRATION.md#1-sensor-placement).

- Dashboard: Tutorial tab step 3 has an interactive top-down room diagram for
  1/2/3-sensor layouts.
- CLI: write your `nodes` list into a config
  ([DESIGN.md](DESIGN.md#config-schema-configjson)) and confirm each sensor
  delivers frames with `marionette doctor`.

## Step 4 - See yourself

Start the pipeline and watch the live fused body:

```sh
marionette run -c config.json
```

- Dashboard: the **Skeleton** tab shows front and top-down views, sensor
  positions, and the tracker diamonds being streamed out. If the skeleton is
  solid while you turn in place, the rig is healthy.
- CLI: the periodic status line (`frames in | fused ticks % | trackers out`)
  tells the same story; see the field table in
  [USAGE.md](USAGE.md#zero-hardware-demo). `--verbose` adds debug logging.

No hardware yet? Use `config/demo.json` again - the Skeleton tab renders the
synthetic walker exactly as it would a real body.

## Step 5 - Calibrate (in this order)

1. **Pair alignment** (multi-sensor rigs) - aligns each extra sensor to the
   reference. Stand in the overlap, move your limbs slowly; accept under
   2-3 cm RMSE.
2. **Body model** - a 15-30 s capture of your bone lengths. Redo per person,
   not per session.
3. **Playspace anchor** (PCVR only) - aligns Marionette's world with the
   SteamVR playspace using a tracked controller. The OSC path never needs it.

- Dashboard: the **Calibration** tab runs all three as guided wizards against
  the live pipeline, with progress and RMSE feedback. One job runs at a time.
  Pair results apply to fusion immediately; a new playspace anchor reaches the
  SteamVR bridge on the next start.
- CLI: `marionette calibrate pair --reference <id> --target <id>`,
  `marionette calibrate body`, `marionette calibrate playspace` - each with
  `-c config.json`. Full procedure and acceptance criteria:
  [CALIBRATION.md](CALIBRATION.md).

## Step 6 - Connect your headset

**Quest (VRChat OSC):**

1. Find the headset IP: Settings > Wi-Fi > your network > details. A static
   DHCP lease helps.
2. Point the `osc` endpoint at it in your config: host = the headset IP, port
   9000.
3. In VRChat: Action Menu > Options > OSC > Enabled.
4. Run Marionette, then run VRChat's FBT calibration (stand straight, arms
   out, confirm).

**SteamVR:**

1. Register the driver once: `scripts/install-driver.ps1` (or `.sh`).
2. Add the `openvr` endpoint (`127.0.0.1:24190`) to your config.
3. Start SteamVR via ALVR / Virtual Desktop / Link, then run Marionette.
   Trackers appear as `MN-WAIST`, `MN-LFOOT`, and so on.
4. Assign roles: SteamVR > Settings > Controllers > Manage Trackers (remembered
   per serial).
5. Quest 2: lower the video bitrate in ALVR / Virtual Desktop - tracker data
   is tiny, the video stream is what competes for bandwidth.

- Dashboard: Tutorial tab step 6 shows both checklists and flags a build
  without OpenVR client support.
- CLI: endpoint config snippets and the full walkthroughs are in
  [USAGE.md](USAGE.md#quest-standalone-vrchat-osc) and
  [USAGE.md](USAGE.md#pcvr-steamvr).

## Step 7 - Done

Trackers should now sit on your body and stay there. Recalibrate only when
something changes:

| Event | Redo |
|---|---|
| Sensor moved, bumped, or remounted | Pair (that sensor) |
| Different person using the rig | Body model |
| SteamVR room setup / playspace recenter / different streaming app | Playspace anchor |
| Trackers drift or sit offset | Anchor first; if turning doubles limbs, Pair |

- Dashboard: keep an eye on the **Events** tab - warnings (a restarting node,
  an outlier-flagged sensor) surface there first.
- CLI: the same events appear in the terminal log; `marionette doctor` is the
  go-to when something feels off.

Full references: [USAGE.md](USAGE.md), [CALIBRATION.md](CALIBRATION.md),
[DESIGN.md](DESIGN.md).
