# Marionette design

Multi-Kinect full-body tracking for SteamVR and Quest. This document is the
source of truth for conventions and component contracts; headers under
`core/include/mn/` are the source of truth for APIs.

## Frames and conventions

- Units: meters, seconds (monotonic, `mn::nowSeconds()`), radians internally.
- **World frame**: right-handed, +Y up, -Z forward (OpenVR convention). By
  default the world frame coincides with the *reference sensor's* local frame
  until a playspace anchor is calibrated.
- **Node-local frame**: right-handed, +Y up, +Z pointing from the sensor
  toward the user (Kinect camera-space convention). Every capture backend must
  emit this, normalizing whatever its SDK uses.
- **Extrinsic** (per node): `Pose` mapping node-local -> world.
- **World anchor**: `Pose` mapping Marionette world -> SteamVR playspace.
  Applied only by the OpenVR bridge. The OSC path needs no anchor: VRChat
  aligns trackers itself using the Head reference we send.
- Device/tracker orientation: local +X right, +Y up, -Z forward.
- Quaternions serialize as `[w,x,y,z]`; vectors as `[x,y,z]`.

## Dataflow

```
capture threads (one per node, node-local frames)
   └─> FusionEngine.submit(nodeId, frame)          [thread-safe mailbox]
tick thread @ cfg.tick_hz:
   FusionEngine.fuse(now) -> world SkeletonFrame
   └─> TrackerMapper.map() -> [TrackerPose]
        ├─> OSC endpoint        (VRChat OSC Trackers, Unity coords)
        └─> OpenVR bridge ──UDP──> driver_marionette ──> SteamVR
```

`Pipeline` (core) owns construction, threading, and stats. Capture nodes and
endpoints are plugins resolved by string type via `NodeRegistry` /
`EndpointRegistry`; the app registers whatever was compiled in.

## Fusion (tier 1)

Per joint j and node n with fresh frame (age <= stale_seconds):

```
w(n,j) = stateW * confidence * depthW * occlusionW
stateW     = 1.0 (Tracked) | inferred_weight (Inferred) | 0 (NotTracked)
depthW     = 1 / (1 + (range_n(j) / depth_noise_ref_m)^2)
occlusionW = occlusion_penalty if j is on the far side of the torso plane
             from sensor n (torso facing estimated from n's own frame), else 1
```

Fused position = weighted mean of world-transformed samples, then per-joint
One-Euro filter, then (optional) bone-length constraint projecting each child
joint onto the calibrated bone length toward its parent (single top-down
pass from Hips). Joint orientations from sensors are discarded; tracker
orientations are derived from fused positions in `TrackerMapper`.

Multiple opposed viewpoints structurally remove the two classic single-Kinect
failures: front/back flip ambiguity and self-occlusion dropouts.

Tier 2 (roadmap): EKF over root pose + joint angles, per-sensor measurement
covariances from the same noise model, constant bone lengths as hard state.

## Outlier rejection

Cross-node guard against a miscalibrated or glitching sensor hijacking a
joint (`fusion.outlier_rejection`, on by default). Per joint, when >= 2 nodes
contribute, each candidate sample is compared against the **weighted
consensus of the OTHER contributors**:

```
consensus_others(i) = sum_{k != i} w_k * worldPos_k / sum_{k != i} w_k
outlier(i)          = |worldPos_i - consensus_others(i)| > outlier_threshold_m
```

Flagged samples get their weight multiplied by `outlier_weight_factor`
(default 0.05) rather than dropped, so a joint never loses all contributors.
The pass is greedy-robust: it repeatedly downweights the single worst
offender and re-checks the rest, so one large glitch cannot drag the
consensus and condemn the good views with it.

**Symmetric two-node caveat**: with exactly two contributors there is no
majority - a disagreement flags BOTH samples (symmetric mistrust), and since
both weights scale by the same factor the fused mean stays between them. The
mechanism only *identifies* the bad sensor with three or more views;
two-node rigs still benefit because the warning event fires on the
transition (once per node-joint, not per tick), pointing at the pair to
recalibrate. Single-node joints are untouched.

## Calibration

1. **Pair (sensor-to-sensor)**: user stands in the overlap; matched joint
   positions across near-simultaneous frames feed Kabsch/Umeyama (`solveRigid`)
   giving target-local -> reference-local; chained through the reference
   node's extrinsic. CLI: `marionette calibrate pair`.
2. **Playspace anchor**: hold a SteamVR-tracked controller in the tracked
   hand, move around; sample (fused wrist world pos, controller SteamVR pos)
   pairs; `solveAnchor` gives world -> SteamVR (the OpenVR-SpaceCalibrator
   move). CLI: `marionette calibrate playspace` (needs OpenVR client lib).
3. **Body model**: per-bone median lengths over a short capture
   (`BodyModelEstimator`), SlimeVR-style.

## Wire protocol (core -> SteamVR driver)

`mn/protocol.hpp`: packed little-endian UDP packets on 127.0.0.1:24190.
`WirePacket{magic 'MNTP', version, count, timestamp, WirePose[count]}`;
`WirePose{role, valid, pos[3], quat wxyz, vel[3], angvel[3]}`. count=0 is a
heartbeat. The driver lazily creates a `TrackedDeviceClass_GenericTracker`
per role on first sight (serials `MN-WAIST`, `MN-LFOOT`, ...), marks poses
invalid after 0.5 s of silence, and feeds velocities so SteamVR predicts
through pipeline latency.

## Config schema (config.json)

```jsonc
{
  "tick_hz": 90,
  "calibration_file": "calibration.json",
  "nodes": [ { "id": "front", "type": "mock|replay|kinect_v2|kinect_v1",
               "params": { /* backend-specific, may carry default "extrinsic" */ } } ],
  "fusion": { "stale_seconds": 0.15, "inferred_weight": 0.25,
              "min_confidence": 0.05, "depth_noise_ref_m": 4.0,
              "occlusion_penalty": 0.3, "bone_length_constraint": true,
              "outlier_rejection": true, "outlier_threshold_m": 0.35,
              "outlier_weight_factor": 0.05,
              "filter": { "min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0 } },
  "mapping": { "trackers": ["waist","left_foot","right_foot","chest",
                            "left_knee","right_knee","left_elbow","right_elbow"],
               "emit_head": true, "velocity_smooth": 0.5 },
  "watchdog": { "enable": true, "silent_seconds": 5.0, "backoff_seconds": 5.0,
                "max_restarts": 10, "exclude_types": ["replay"] },
  "dashboard": { "enable": true, "bind": "127.0.0.1", "port": 8211 },
  "endpoints": [ { "type": "osc",    "params": { "host": "<quest-ip>", "port": 9000 } },
                 { "type": "openvr", "params": { "host": "127.0.0.1", "port": 24190 } } ]
}
```

All fusion/mapping/watchdog/dashboard keys optional (defaults above).
calibration.json:
`{ "extrinsics": { "<nodeId>": Pose }, "body_model": { "valid": bool,
"bone_lengths": { "<childJointName>": meters } }, "world_anchor": Pose }`.

## Sensor backend notes

- **Kinect v2 / SDK 2.0** (Windows): one sensor per PC (hard SDK limit),
  25 joints, `CameraSpacePoint` is right-handed +Y up +Z toward user - already
  node-local convention. No per-joint confidence: Tracked=1.0, Inferred=0.5.
- **Kinect v1 / SDK 1.8** (Windows): several sensors per PC, 20 joints
  (synthesize Chest as midpoint(spine, neck-ish) per mapping table in the
  backend). Structured-light: two v1s with overlapping views degrade each
  other; v1+v2 mix is clean (different modalities).
- **libfreenect2 markerless** (roadmap): many v2s on one box (separate
  NEC/Intel PCIe USB3 controllers, ~1 controller per sensor), RGB+registered
  depth into a 2D keypoint model (RTMPose/BlazePose) lifted to 3D by depth.
- **Remote nodes** (roadmap): same SkeletonFrame stream over LAN from capture
  PCs hosting extra v2s.

## Threading and failure model

One thread per capture node (owned by the backend), one tick thread, endpoints
are called on the tick thread and must not block (UDP fire-and-forget). A node
crashing or going stale only removes its rows from fusion - the rig degrades
gracefully. All cross-thread handoff happens in `FusionEngine.submit` (mutex'd
latest-frame mailbox per node).

## Watchdog

`Pipeline` monitors per-node frame recency on the tick thread
(`cfg.watchdog`, on by default). A node silent past `silent_seconds` - and
not of an excluded type (`exclude_types`, default `["replay"]`, which is
legitimately finite) - is **recreated from its registry factory** and
restarted: the old `ICaptureNode` is stopped and destroyed, a fresh instance
is built from the same `NodeConfigEntry` via the `NodeRegistry`, and
`start()` is called again. Restart attempts are rate-limited to one per
`backoff_seconds` per node and capped at `max_restarts` per node per run;
a node that exhausts its budget stays down (the rig keeps running without
it). Restart counts surface in `Pipeline::nodeStatuses()` (dashboard
Home tab) and each restart logs a warning into the event log.

Lifetime consequence: because the watchdog re-invokes factories at any point
during a run, **the `NodeRegistry` and `EndpointRegistry` passed to
`Pipeline::build()` must outlive the Pipeline** (contract on
`mn/pipeline.hpp`).

## Events

`mn::EventLog` (`mn/events.hpp`) is a global, thread-safe ring buffer of the
most recent 2000 events: `{seq, t, level, message}` with `seq` monotonically
increasing from 1. `EventLog::installLogCapture()` routes every
`mn::log::write()` into the ring via `log::setSink` (the app installs it
first thing in `main`, before anything can log); subsystems can also `push()`
directly. Consumers: the dashboard (`GET /api/events?after=SEQ` - clients
poll with their last seen `seq`) and `marionette doctor`. Lifetime per-level
totals (`levelCounts()`) are not capped by the ring, so warn/error counters
on the dashboard stay accurate after wraparound. The sink replaces any
previous one (no composition); `clear()` drops buffered events but `seq`
keeps rising.

## Dashboard

`mn::dash::DashboardServer` (`server/`, cpp-httplib, `MN_BUILD_DASHBOARD`)
serves a **single embedded HTML app** (`web/index.html`, baked into the
binary at build time; `Options::webDirOverride` serves from disk for the
frontend dev loop) plus the JSON API in
[DASHBOARD_API.md](DASHBOARD_API.md). It binds loopback by default, no auth;
binding `0.0.0.0` is an explicit config choice.

Integration with the pipeline is read-mostly and observer-based:

- Status/skeleton endpoints read `Pipeline::stats()`, `nodeStatuses()`,
  `latestFused()`, `latestTrackers()` - all snapshot getters, no locking the
  tick loop.
- For calibration the dashboard owns the pipeline's **single observer slots**
  (`setRawFrameObserver` / `setFusedFrameObserver`, replace semantics): raw
  frames feed pair sessions, fused frames feed body/playspace sampling.
- Calibration jobs run on an internal worker thread, **at most one at a
  time** (`POST` while busy returns `{"ok":false,"error":"busy"}`,
  `/api/calibrate/cancel` aborts). Successful jobs persist the
  `CalibrationStore`; pair jobs additionally call
  `Pipeline::applyNodeExtrinsic` so the fix is live without a restart, while
  a new playspace anchor reaches the SteamVR bridge on the next pipeline
  start (anchor injection happens at endpoint build time).
- The playspace job itself is app-supplied (`Options::playspace`): only the
  app links the OpenVR client lib, so the server stays free of that
  dependency and returns 501 when the hook is absent.
