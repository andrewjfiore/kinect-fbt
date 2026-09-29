# Marionette User Guide

Everything you need to go from an unopened box of Kinect sensors to full-body
tracking in VR - in plain language, no command line required.

- New here? Read this top to bottom once.
- Just want to install? Jump to [INSTALL.md](INSTALL.md).
- Want the terse reference? See [USAGE.md](USAGE.md).

**Contents**

1. [What Marionette is](#1-what-marionette-is)
2. [Install it](#2-install-it)
3. [The dashboard](#3-the-dashboard)
4. [First-time setup, step by step](#4-first-time-setup-step-by-step)
5. [Placing your sensors](#5-placing-your-sensors)
6. [Checking projection health (mirrored / backwards / upside-down fixes)](#6-checking-projection-health)
7. [Calibration](#7-calibration)
8. [Connecting your headset](#8-connecting-your-headset)
9. [Keeping it healthy](#9-keeping-it-healthy)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What Marionette is

Marionette turns one or more Microsoft Kinect sensors into full-body trackers
for VR. Each sensor watches you from a different angle; a fusion engine merges
every view into one stable skeleton and streams tracker poses to your game.

Because several viewpoints see you at once, Marionette avoids the two classic
single-Kinect failures: the skeleton flipping front-to-back, and limbs
vanishing when your body hides them from the camera.

It sends trackers two ways, and you can use either or both:

- **Quest standalone** - straight to the headset over Wi-Fi using VRChat's OSC
  trackers. No gaming PC stream required.
- **PCVR / SteamVR** - as virtual SteamVR trackers, for use with ALVR, Virtual
  Desktop, Steam Link, or Quest Link.

You do not need a Kinect to look around first: Marionette ships with a built-in
demo that runs two simulated sensors through the real pipeline, so the whole
app works on day one.

## 2. Install it

Download the installer for your system and double-click it - full steps in
[INSTALL.md](INSTALL.md). In short:

- **Windows** - run `Marionette-<version>-Setup.exe`, then launch **Marionette**
  from the Start Menu.
- **Linux** - download the `.AppImage`, mark it executable, and double-click.

When you launch Marionette it starts quietly in the background and opens its
**dashboard** in your web browser. That dashboard is where you do everything.
There is no console window and nothing to type.

## 3. The dashboard

The dashboard is a local web page (at `http://127.0.0.1:8211`) that only your
computer can see. It has five tabs:

| Tab | What it is for |
|---|---|
| **Overview** | Is everything running? Sensor cards show frame rate, whether they see a body, and any errors. |
| **Skeleton** | Watch your live tracked body from the front and top. Also where you check and fix projection accuracy. |
| **Calibration** | Guided wizards that tune the system to your room and your body. |
| **Events** | A running log of warnings and errors, newest first. |
| **Tutorial** | The same walkthrough as this guide, built in and interactive. It opens by itself the first time. |

On your first visit the **Tutorial** tab greets you and walks you through the
whole setup. This guide covers the same ground with a bit more depth.

## 4. First-time setup, step by step

1. **Launch Marionette.** The dashboard opens. If the **Overview** tab shows a
   sensor with a green frame rate and "body: yes" when you step in front of it,
   your sensor is working. (No sensor yet? The demo shows a walking figure so
   you can still explore.)
2. **Pick how you'll connect** - Quest over Wi-Fi, or PCVR through SteamVR. See
   [section 8](#8-connecting-your-headset). You can set up one now and the other
   later.
3. **Place your sensors** - [section 5](#5-placing-your-sensors).
4. **Open the Skeleton tab** and check that your body looks right. If it faces
   the wrong way or your left and right are swapped, fix it in
   [Projection health](#6-checking-projection-health) - one click.
5. **Calibrate** - [section 7](#7-calibration).
6. **Connect your headset** and run your game's tracker calibration.

## 5. Placing your sensors

- Mount each sensor at roughly head height (about 1.6-2.0 m), tilted slightly
  down, with the middle of your play area 2-3.5 m away.
- Two sensors cover most of a full turn. Place them **opposite** each other
  (front and back) or at a **right angle** (front and side).
- Only **one Kinect v2** can run per PC (a hardware limit). You can use several
  Kinect v1s, but never point two v1s at the same space - their infrared
  patterns interfere. Mixing a v1 with a v2 is fine.
- Mount sensors firmly. If one gets bumped, redo that sensor's pair calibration.

More detail and diagrams: [CALIBRATION.md](CALIBRATION.md#1-sensor-placement),
and the Tutorial tab has an interactive top-down room layout.

## 6. Checking projection health

Open the **Skeleton** tab and stand where your sensors can see you. Below the
live view is the **Projection health** panel. It does two jobs.

**It checks that points are projected accurately.** On every frame it verifies
your tracked body is finite, sits within a sensible distance of the centre, has
believable limb lengths, and is the right way up. The status chip reads **OK**
when all is well, or **Check** with a short reason when something is off. The
four readouts (farthest joint, worst bone error, head-above-hips, tracked
joints) tell you *what* is wrong, and the offending one turns red.

**It lets you correct axis errors in one click.** Kinect rigs commonly come out
mirrored, backwards, or (rarely) upside down depending on how a sensor is
mounted. Four switches fix it, instantly and saved for next time:

| If your body... | Turn on |
|---|---|
| faces the wrong direction (turned 180 degrees) | **Flip X** and **Flip Z** |
| has left and right swapped (a mirror image) | **Flip X** and **Swap L / R** |
| appears upside down | **Flip Y** |

Watch the skeleton as you toggle - the right combination snaps it into place and
the chip returns to **OK**. Flip Z alone reverses front/back; Flip X alone
mirrors left/right in space; Swap L/R relabels which side is which. Most people
only ever need one of the recipes above.

## 7. Calibration

Do these in order, on the **Calibration** tab. Each is a guided wizard with a
progress bar; only one runs at a time.

1. **Pair alignment** (only if you have more than one sensor) - teaches the
   system how your sensors are positioned relative to each other. Stand where
   both see you and move your arms and legs slowly. Aim for under 2-3 cm error.
2. **Body model** - a short capture (15-30 seconds) of your bone lengths. Redo
   it for each different person, not each session.
3. **Playspace anchor** (PCVR only) - lines Marionette's world up with your
   SteamVR play space. Hold a tracked controller and sweep it around. The Quest
   Wi-Fi path never needs this.

Full procedure and what "good" looks like: [CALIBRATION.md](CALIBRATION.md).

## 8. Connecting your headset

### Quest over Wi-Fi (simplest)

1. Find your headset's IP address (Settings -> Wi-Fi -> your network -> details).
2. In your Marionette config, point the `osc` output at that IP on port 9000.
3. In VRChat: Action Menu -> Options -> OSC -> **Enabled**.
4. Run Marionette, then run VRChat's own full-body calibration (stand straight,
   arms out, confirm).

### PCVR through SteamVR

1. Make sure the **SteamVR driver is registered** - the Windows installer offers
   this during setup; on Linux the tarball installer does it, or run the bundled
   `install-driver` helper once.
2. Add the `openvr` output to your config.
3. Start SteamVR (through ALVR / Virtual Desktop / Link), then run Marionette.
   Trackers appear named `MN-WAIST`, `MN-LFOOT`, and so on.
4. In SteamVR, assign each tracker to a body role (Settings -> Controllers ->
   Manage Trackers). It remembers them next time.

Step-by-step config snippets for both paths:
[USAGE.md](USAGE.md#quest-standalone-vrchat-osc) and
[USAGE.md](USAGE.md#pcvr-steamvr).

## 9. Keeping it healthy

You only need to recalibrate when something changes:

| What changed | Redo |
|---|---|
| A sensor moved or got bumped | Pair alignment (that sensor) |
| A different person is using it | Body model |
| SteamVR room setup / recenter / different streaming app | Playspace anchor |
| Trackers drift or sit offset | Playspace anchor first; if turning doubles your limbs, Pair |

Keep an eye on the **Events** tab - warnings such as a sensor restarting or a
view being flagged as an outlier show up there first. The **Overview** tab is
your at-a-glance health check.

## 10. Troubleshooting

**The dashboard didn't open.** It is always at `http://127.0.0.1:8211` -
open that in any browser. If it still won't load, Marionette may not be running;
launch it again.

**A sensor isn't seen** (Overview shows no frames). On Windows, install the
matching Kinect SDK runtime so the operating system recognizes the sensor
([SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561) for
v2, [1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) for
v1). Check the USB and power connections. On Windows you can also run the
bundled `scripts\fix-kinect-services.ps1` helper.

**My body is mirrored / backwards / upside down.** That's a projection error -
fix it with the switches in [Projection health](#6-checking-projection-health).

**Trackers drift or sit off my body.** Redo the playspace anchor. If turning in
place makes your limbs split into two, redo pair calibration.

**SteamVR trackers don't appear.** Confirm the driver is registered (re-run the
installer or the `install-driver` helper), that SteamVR is running, and that the
`openvr` output is in your config. Restart SteamVR after registering.

**Quest 2 tracking feels laggy.** Lower the video bitrate in ALVR / Virtual
Desktop. Tracker data is tiny; the video stream is what competes for Wi-Fi.

Still stuck? The **Events** tab and [USAGE.md](USAGE.md) go deeper, and the
built-in health check (`doctor`) is described in
[USAGE.md](USAGE.md#doctor).
