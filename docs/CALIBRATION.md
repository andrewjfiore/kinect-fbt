# Calibration

Three independent calibrations, all stored in the config's `calibration_file`
(default `calibration.json`): sensor **pair** extrinsics, the per-user **body
model**, and the SteamVR **playspace anchor**. Do them in that order.

Every step can be run two ways: the CLI commands shown below, or the
**dashboard Calibration tab** while `marionette run` is active
([USAGE.md](USAGE.md#dashboard)) - same solvers, same acceptance criteria,
with live progress and RMSE in the browser. The dashboard runs one
calibration job at a time.

Build/run basics: [BUILD.md](BUILD.md), [USAGE.md](USAGE.md). Frame
conventions and the math behind each step: [DESIGN.md](DESIGN.md#calibration).

## 0. Pre-flight: doctor

Before calibrating, run the environment check:

```sh
marionette doctor -c config.json
```

It verifies every sensor actually delivers frames, the config's node types
exist in this build, the calibration file parses, and (for the PCVR path)
that the SteamVR driver is registered. Fix anything it lists before blaming a
calibration. Details and sample output: [USAGE.md](USAGE.md#doctor).

## 1. Sensor placement

- Mount sensors at roughly **head height** (1.6-2.0 m), tilted slightly down,
  with the play area center 2-3.5 m away. Kinect depth noise grows roughly
  quadratically with range; past ~4 m a sensor contributes little weight.
- **Two sensors cover most of 360 degrees.** Place them **opposing** (front +
  back) or at a **90-degree offset** (front + side). Opposed viewpoints are
  what structurally removes the classic single-Kinect failures (front/back
  flip, self-occlusion), so prefer placements where at least two sensors see
  you from meaningfully different directions.
- Every added sensor needs view overlap with the reference (or an
  already-calibrated sensor) - pair calibration aligns sensors through a body
  both can see at once.
- **Do not overlap two Kinect v1s.** v1 is structured-light: two v1s
  projecting IR dot patterns into the same scene degrade each other badly.
  A **v1 + v2 mix is clean** (structured light vs time-of-flight, different
  modalities), as is v2 + v2.
- The Microsoft SDKs allow **one v2 per PC** (hard SDK limit; v1s can share a
  PC). For the future multi-v2-per-box path (libfreenect2, see ROADMAP),
  plan on roughly **one USB3 controller per v2** - separate NEC/Intel PCIe
  USB3 cards, not one card's ports.
- Mount rigidly. Pair calibration is only valid while the sensors do not
  move - a bumped tripod means recalibrating that sensor.

## 2. Pair calibration (sensor-to-sensor)

Solves each target sensor's pose relative to the reference sensor by matching
joint positions across near-simultaneous frames (Kabsch/Umeyama). The
reference sensor's frame is the world frame until a playspace anchor exists.

1. Start it: `marionette calibrate pair -c config.json --reference <id>
   --target <id>` (both required; `--min-samples` and `--max-seconds` are
   optional, defaults 200 and 60). With more than two nodes, calibrate each
   target against the reference (or an already-calibrated sensor) one at a
   time. Or: dashboard **Calibration tab > Pair alignment**, pick reference
   and target, Start.
2. Stand in the **overlap region** - both sensors must track your full body
   at once.
3. **Move your limbs**, slowly: raise and lower each arm through a wide arc,
   lift each foot, do a slow half-turn, take a step around the overlap. The
   solver matches head, hips, wrists, and ankles; it needs them confidently
   tracked in both views and spread out in space (varied, non-collinear
   samples). Avoid fast motion - frames are paired within a 50 ms window, so
   speed turns into pairing error.
4. The session accumulates matched point pairs (at least ~200 needed) and then
   solves, reporting the **RMSE** residual in meters.

**Accept under 2-3 cm RMSE.** Joint-fusion alignment does not need to be
millimeter-perfect; the One-Euro filter and weighting absorb the rest. If it
is worse:

- You moved too fast, or stood at the edge of a sensor's range. Repeat slower
  and closer.
- Samples were too clustered (standing still in one pose). Cover more of the
  overlap volume.
- A v1/v1 overlap is fighting itself (see placement above).

The solved extrinsic is written to `calibration.json` under `extrinsics`.
A pair job run from the dashboard also **applies the new extrinsic to the
running fusion engine immediately** - no restart needed; the CLI path takes
effect on the next `marionette run`. Verify by watching the fused output
(easiest on the dashboard Skeleton tab) stay solid as you turn in place: a
bad pair calibration shows up as limbs splitting into doubles when you rotate
between views.

## 3. Body model (bone lengths)

Estimates your per-bone lengths (per-bone median over a short capture,
SlimeVR-style). Enables `fusion.bone_length_constraint`, which projects each
joint onto the calibrated bone length and suppresses depth-noise stretch.

1. `marionette calibrate body -c config.json` (`--seconds <sec>` optional,
   default 15). Or: dashboard **Calibration tab > Body model**, set the
   capture length, Start.
2. Stand fully in view of your best sensor (or the calibrated rig), arms
   slightly away from your body. Move gently - shift weight, bend each knee
   and elbow a little - for the duration of the capture (around 15-30 s).
3. The result is written to `calibration.json` under `body_model`.

Calibrate per person. Footwear and clothing changes do not matter; a
different user does.

## 4. Playspace anchor

Aligns Marionette's world with the **SteamVR playspace** (the
OpenVR-SpaceCalibrator move). Only the `openvr` endpoint uses it - the OSC /
Quest-standalone path never needs this step, because VRChat aligns trackers to
the head reference itself.

Requires: a build with the OpenVR client lib (`MN_WITH_OPENVR_CLIENT`, the
default; see [BUILD.md](BUILD.md#cmake-options)), SteamVR running, and a
tracked controller.

1. Start SteamVR (via ALVR / Virtual Desktop / Link / wired headset).
2. `marionette calibrate playspace -c config.json` (`--hand left|right` and
   `--seconds <sec>` optional, defaults right and 20). Or: dashboard
   **Calibration tab > Playspace anchor**, pick the hand, Start - this
   samples against the live pipeline while it keeps running.
3. Hold a SteamVR controller firmly in one hand and keep that wrist visible to
   the sensors. The solver pairs the fused wrist position (Marionette world)
   with the controller position (SteamVR).
4. **Move the hand through the volume**: wide slow arcs, high and low, across
   the play area - not along a single line. Degenerate (collinear) coverage
   makes the solve ill-conditioned.
5. The anchor is written to `calibration.json` under `world_anchor`. Either
   way it is run, the new anchor reaches the SteamVR bridge **on the next
   pipeline start** - restart `marionette run` for trackers to move (the
   dashboard job's result message says the same).

Verify in SteamVR: the trackers should sit on your body, and stay there as you
walk to the edges of the playspace. An offset that grows with distance from
center means a bad anchor - redo step 4 with wider coverage.

## When to recalibrate

| Event | Redo |
|---|---|
| A sensor was moved, bumped, or remounted | Pair (that sensor) |
| Sensor added or removed | Pair (new/affected sensors) |
| Different person using the rig | Body model |
| SteamVR room setup / playspace recenter / different streaming app | Playspace anchor |
| Trackers drift or sit offset in SteamVR | Playspace anchor first; if turning in place doubles limbs, Pair |

Pair extrinsics and the body model are stable across sessions as long as
nothing physically changed. The playspace anchor is the most fragile of the
three - some streaming stacks re-seat the playspace per session, so re-running
`calibrate playspace` at session start is cheap insurance.

## Projection correction (mirrored / backwards / upside-down)

Calibration fixes *where* your sensors are; it does not fix a whole-body
**handedness or facing error** - a rig mounted so the fused skeleton comes out
mirrored, turned 180 degrees, or upside down. That is a projection correction,
not a calibration, and it lives on the dashboard's **Skeleton** tab under
**Projection health** (full reference:
[USAGE.md](USAGE.md#projection-health-and-axis-correction)).

That panel also runs a live **validity check** on the projected skeleton
(finite positions, in-bounds joints, plausible bone lengths, upright), so a bad
extrinsic that flings a joint metres away shows up immediately rather than as
silently wrong trackers.

The correction (Flip X / Y / Z, Swap L/R) is saved in the calibration file under
`"projection"`, so it persists with the rest of your calibration:

```jsonc
"projection": { "flip_x": false, "flip_y": false, "flip_z": false, "swap_lr": false }
```
