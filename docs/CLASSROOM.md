# Marionette for classrooms and labs

Marionette works anywhere someone can plug a Kinect into a PC: a student VR
club, a university movement lab, a medical school's rehab or anatomy
teaching space, a makerspace, a physio clinic's demo corner. This page is
the shortest path from "we have a room" to "students are watching a live
skeleton", with the budget, privacy, and logistics questions answered up
front.

No VR headset is required for most of what's below. The live skeleton view,
calibration, and recording all run on a plain PC and a wall of curiosity.

## What it costs

Kinects were mass-produced for a decade and the used market is deep. Prices
are rough 2026 second-hand figures; check your local listings.

| Item | Roughly | Notes |
|---|---|---|
| Kinect v1 (Xbox 360) | $10-25 | Plus a 12 V USB adapter (~$10) if it didn't come with one |
| Kinect v2 (Xbox One) | $25-50 | Plus the USB 3 / power adapter kit (~$20). Better skeleton than v1 |
| Tripod or wall mount | $15-25 each | Anything rigid works; rigid matters more than fancy |
| PC | whatever you have | A mid-range machine from the last decade is fine; no GPU needed |

A complete two-sensor station lands around $100-150. A single-sensor
demo station can be under $40.

Hardware capture currently runs on **Windows** (the sensors talk through
Microsoft's free Kinect SDKs; a cross-platform backend is on the roadmap).
The zero-hardware demo, the dashboard, and the whole build run on Windows
and Linux.

## Nothing leaves the room

Worth stating plainly, because it decides whether an institution can say
yes:

- Marionette needs **no account, no login, and no internet** to run. After
  the one-time SDK download, an offline machine works fine.
- The sensors' video is consumed inside the capture pipeline and thrown
  away. What flows through Marionette is **joint positions**: ~19 labeled
  points in space, no images.
- **Nothing is recorded** unless someone explicitly runs
  `marionette record`, and what that writes is the same joint data, as a
  local JSONL file you can open in a text editor.
- The dashboard is served to **that PC only** (`127.0.0.1`) unless you
  deliberately open it to the local network.

If your institution requires consent forms for motion recording, the honest
summary is: "a depth camera estimates the positions of 19 body joints;
images are not stored or transmitted."

## A one-hour setup, once

1. **Build or copy the binary** ([BUILD.md](BUILD.md)). One person does
   this once; the result is a single folder you can copy to every station.
2. **Install the Kinect SDK** for your sensor version (free from
   Microsoft; needs admin rights, so line up IT beforehand).
3. **Mount the sensors** and run the 15-minute walkthrough
   ([TUTORIAL.md](TUTORIAL.md), or the dashboard's built-in Guide tab).
4. **Save the config and calibration files.** Sensor alignment sticks as
   long as nothing moves, so a taped-down tripod means step 3 happens once
   per room, not once per class.

Before each session, `marionette doctor` tells you in ten seconds whether
the station is healthy, in plain text with an `OK` at the bottom.

Per-person setup is one step: the body measurement capture, about 15
seconds of standing in view (dashboard: Calibrate tab > Body
measurements). Swapping people takes under a minute.

## Things to do with it

- **See yourself as a skeleton.** The dashboard's Skeleton tab is the
  reliable crowd-pleaser and needs zero extra equipment. Walk, squat,
  wave; watch confidence drop when limbs occlude and recover when a second
  sensor fills in. That last part quietly teaches sensor fusion.
- **Watch movement, together.** Gait, balance, sit-to-stand, reach: a
  live top-down and front view that a whole group can see at once makes
  movement patterns discussable in a way that watching the person directly
  doesn't.
- **Record and analyze.** `marionette record` writes time-stamped joint
  positions to JSONL, which loads into Python or a spreadsheet in a few
  lines. Joint angles over time, stride timing, range of motion: all a
  short script away. The `replay` node plays recordings back through the
  live pipeline, so one recording can drive a whole lab section.
- **Full-body VR.** With a headset, the full setup from
  [TUTORIAL.md](TUTORIAL.md) gives students legs in VRChat or any SteamVR
  game that takes trackers, which is its own lesson in coordinate frames,
  calibration, and latency.

One honest boundary: this is consumer depth-camera tracking, accurate to a
few centimeters on a good day. It is great for teaching, demonstration, and
exploratory work, and it is **not a medical device**; don't base clinical
decisions on it.

## When something breaks mid-class

- The dashboard's Home tab has a setup checklist; anything unchecked is
  the problem, and the `fix` link goes to the tab that solves it.
- A sensor that goes silent is restarted automatically by the watchdog;
  you'll see it in the Log tab rather than as a mystery freeze.
- `marionette doctor` is the ten-second triage for everything else:
  cables, config, ports, missing SDKs.
- Someone bumped a tripod: redo alignment for that one sensor (two
  minutes). Nothing else is lost.

Questions the docs don't answer are welcome in the repo's issue tracker.
