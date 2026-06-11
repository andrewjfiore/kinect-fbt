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
              "filter": { "min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0 } },
  "mapping": { "trackers": ["waist","left_foot","right_foot","chest",
                            "left_knee","right_knee","left_elbow","right_elbow"],
               "emit_head": true, "velocity_smooth": 0.5 },
  "endpoints": [ { "type": "osc",    "params": { "host": "<quest-ip>", "port": 9000 } },
                 { "type": "openvr", "params": { "host": "127.0.0.1", "port": 24190 } } ]
}
```

All fusion/mapping keys optional (defaults above). calibration.json:
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
