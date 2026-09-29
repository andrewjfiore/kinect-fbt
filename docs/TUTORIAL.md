# Tutorial: from an empty room to trackers on your body

This is the standalone copy of the dashboard's built-in guide (the Guide
tab). It assumes nothing: no prior setup, no calibration, no familiarity with
the project. Each step names the dashboard tab that does the job and the CLI
command that does the same thing without a browser.

You need a built binary ([BUILD.md](BUILD.md)). Hardware is optional until
step 5: everything before that works against the zero-hardware demo config.

Quick orientation: `marionette run` serves a local web dashboard at
**http://127.0.0.1:8211** (when built with `MN_BUILD_DASHBOARD`, the
default). Its tabs are **Home** (status headline, setup checklist, sensor
cards), **Skeleton** (live fused body), **Calibrate** (guided wizards for all
three calibrations), **Log** (warnings and errors), and **Guide** (this
walkthrough, interactive). See [USAGE.md](USAGE.md#dashboard) for the full
tab reference.

## Step 1 - What Marionette does

Marionette turns Kinect sensors, the old Xbox cameras, into full-body
tracking for VR. Point one or two at your play space and your hips, knees,
and feet follow you into the game. Each sensor watches you from a different
angle; Marionette merges the views into one solid skeleton and streams it to
SteamVR or straight to a Quest headset over Wi-Fi.

Why more than one sensor? A camera behind you means your skeleton doesn't
flip when you turn around, and your arms don't vanish when they pass behind
your back. One sensor works fine for seated or front-facing play.

**Try it right now, no hardware:**

```sh
marionette run -c config/demo.json
```

This is a practice rig: two pretend sensors (front + back, walking in place,
with simulated depth noise) running through the real fusion pipeline. Open
http://127.0.0.1:8211 and poke around; every step in this tutorial can be
dry-run against it.

- Dashboard: the Guide tab opens automatically on first visit.
- CLI: `marionette -h` lists every subcommand; `marionette list-types` shows
  which sensor and output types are compiled into your binary.

## Step 2 - Pick your path

Two ways to get the trackers into your game. You can run both at once later;
pick one to start with.

| | Quest standalone (easiest) | PCVR / SteamVR (works everywhere) |
|---|---|---|
| How it connects | Straight to VRChat on the headset over Wi-Fi | OpenVR driver via ALVR / Virtual Desktop / Link / cable |
| Needs SteamVR | No | Yes |
| Needs driver install | No | Yes, one time |
| Needs playspace sync | No | Yes |
| Trackers | Up to 8 via OSC; hips + feet is plenty to start | Real SteamVR devices (`MN-*` serials), any PCVR game that supports trackers |

Full setup details for each path: [USAGE.md](USAGE.md#quest-standalone-vrchat-osc)
and [USAGE.md](USAGE.md#pcvr-steamvr).

**Pre-flight check** before touching hardware:

```sh
marionette doctor -c config.json
```

`doctor` checks everything in one go: compiled backends, attached sensors,
your config, the calibration file, ports, and the SteamVR driver. It ends
with `result: OK` or a list of what to fix ([USAGE.md](USAGE.md#doctor)).
The dashboard's Home tab shows a lighter version of the same picture.

## Step 3 - Place your sensors

- Put each sensor at about head height (1.6-2.0 m), tilted slightly down,
  2-3.5 m from where you'll stand. Closer is sharper; past about 4 m a
  sensor can't see enough detail to help.
- Two sensors facing each other cover you from nearly every angle. That's
  the sweet spot. A 90-degree offset (front + side) also works.
- One Kinect v2 per PC: the Microsoft SDK allows exactly one. Older v1
  sensors don't have this limit.
- Don't aim two v1 sensors at the same spot; their infrared patterns
  scramble each other. Mixing a v1 with a v2 is fine, and so is v2 + v2.
- Mount them so they can't move. A bumped tripod means redoing alignment
  for that sensor.

Full placement guidance: [CALIBRATION.md](CALIBRATION.md#1-sensor-placement).

- Dashboard: Guide tab step 3 has an interactive top-down room diagram for
  1/2/3-sensor layouts.
- CLI: write your `nodes` list into a config
  ([DESIGN.md](DESIGN.md#config-schema-configjson)) and confirm each sensor
  delivers frames with `marionette doctor`.

## Step 4 - Calibrate

Quick check first: start the pipeline and watch yourself move.

```sh
marionette run -c config.json
```

- Dashboard: the **Skeleton** tab shows front and top-down views, sensor
  positions, and the tracker diamonds being streamed out. If the figure
  moves when you move, your sensors are working.
- CLI: the periodic status line (`frames in | fused ticks % | trackers out`)
  tells the same story; see the field table in
  [USAGE.md](USAGE.md#zero-hardware-demo).

Then run the calibration steps, in this order:

1. **Sensor alignment** (2+ sensors) - teaches the sensors where they are
   relative to each other. Stand where both can see you and move your arms
   and legs slowly. The dashboard rates the result out of three stars; aim
   for three (under 2 cm of drift), two is fine. One-sensor rigs skip this.
2. **Body measurements** - a 15-30 second capture of your proportions so
   the skeleton fits you. Redo it when a new person uses the rig, not every
   session.
3. **Playspace sync** (SteamVR only) - lines Marionette's world up with
   SteamVR's using a tracked controller. Quest-only setups never need it.

- Dashboard: the **Calibrate** tab runs all three as guided wizards against
  the live pipeline, with progress bars and the star rating. One runs at a
  time. Alignment results apply immediately; a new playspace sync reaches
  the SteamVR bridge on the next start.
- CLI: `marionette calibrate pair --reference <id> --target <id>`,
  `marionette calibrate body`, `marionette calibrate playspace` - each with
  `-c config.json`. Full procedure and acceptance criteria:
  [CALIBRATION.md](CALIBRATION.md).

## Step 5 - Connect and play

**Quest (VRChat):**

1. Find the headset's IP address: Settings > Wi-Fi > your network > details.
   A static DHCP lease helps.
2. In your config, point the `osc` output at that IP, port 9000.
3. In VRChat: Action Menu > Options > OSC > Enabled.
4. Start Marionette, then run VRChat's FBT calibration: stand straight,
   arms out, confirm.

**SteamVR:**

1. Install the driver once: `scripts/install-driver.ps1` (or `.sh` on
   Linux).
2. Add the `openvr` output (`127.0.0.1:24190`) to your config.
3. Start SteamVR via ALVR / Virtual Desktop / Link, then Marionette.
   Trackers show up named `MN-WAIST`, `MN-LFOOT`, and so on.
4. Tell SteamVR where each one goes: Settings > Controllers > Manage
   Trackers. It remembers per serial.
5. Stuttering on Quest 2? Lower the video bitrate in ALVR / Virtual
   Desktop. The trackers themselves use almost no bandwidth; the video
   stream is what competes for it.

That's the whole setup. Later on, recalibrate only when something changes:

| What happened | Redo |
|---|---|
| Sensor moved, bumped, or remounted | Alignment (that sensor) |
| Someone else using the rig | Body measurements |
| SteamVR room setup, playspace recenter, or new streaming app | Playspace sync |
| Trackers drift or sit offset | Playspace sync first; if limbs double when you spin, alignment |

If something acts up, check the dashboard's **Log** tab; problems show up
there first. `marionette doctor` is the go-to when something feels off.

Full references: [USAGE.md](USAGE.md), [CALIBRATION.md](CALIBRATION.md),
[DESIGN.md](DESIGN.md). Setting this up for a group, a lab, or a classroom?
See [CLASSROOM.md](CLASSROOM.md).
