# Dashboard HTTP API

Served by `mn::dash::DashboardServer` (default `http://127.0.0.1:8211`).
All responses are JSON (`Content-Type: application/json`) except `GET /`.
No auth: loopback by default; binding beyond loopback is an explicit choice.
Every POST returns `{"ok": bool, "error": "..."}` at minimum. Unknown routes
404 with `{"ok": false, "error": "not found"}`.

## GET /
The embedded single-file UI (`web/index.html`), `Content-Type: text/html`.

## GET /api/status
```jsonc
{
  "app": { "name": "marionette", "version": "0.1.0", "uptime_s": 12.3,
           "tick_hz": 60.0,
           "capabilities": { "kinect_v2": true, "kinect_v1": true,
                              "openvr_client": true, "dashboard": true } },
  "pipeline": { "running": true, "frames_in": 1234, "ticks": 720,
                "fused_ticks": 715, "fused_pct": 99.3, "trackers_out": 2860 },
  "nodes": [ { "id": "front", "type": "kinect_v2", "running": true,
               "frames": 612, "fps": 29.8, "last_frame_age_s": 0.03,
               "has_body": true, "restarts": 0, "last_error": "" } ],
  "endpoints": [ { "type": "osc" }, { "type": "openvr" } ],
  "trackers": [ { "role": "waist", "valid": true,
                  "pos": [0.01, 0.95, -0.02] } ],
  "calibration": { "file": "calibration.json",
                   "extrinsics": ["front", "back"],   // node ids present
                   "body_model_valid": false,
                   "world_anchor_set": false },
  "events": { "warn": 0, "error": 0, "latest_seq": 42 }
}
```
`fps` and `last_frame_age_s` come from `Pipeline::nodeStatuses()`.

## GET /api/skeleton
Polled at ~15 Hz by the UI.
```jsonc
{
  "t": 123.456, "has_body": true,
  "joints": { "head": { "p": [x,y,z], "c": 0.93, "s": 2 }, ... },  // all 19
  "bones": [ ["head","neck"], ... ],          // child->parent name pairs
  "sensors": [ { "id": "front", "type": "kinect_v2",
                 "pos": [0,0.9,-2], "fwd": [0,0,1] } ], // extrinsic-derived
  "trackers": [ { "role": "waist", "valid": true,
                  "pos": [x,y,z], "rot": [w,x,y,z] } ]
}
```
`s`: 0 NotTracked, 1 Inferred, 2 Tracked. `fwd` = sensor view direction in
world (extrinsic rotation applied to local +Z). When `has_body` is false,
`joints` may be `{}`.

## GET /api/events?after=SEQ
Events with `seq > after` (default 0), oldest first, max 500.
```jsonc
{ "events": [ { "seq": 41, "t": 120.0, "level": "warn", "msg": "..." } ],
  "counts": { "debug": 0, "info": 12, "warn": 1, "error": 0 },
  "latest_seq": 42 }
```
`level` is one of `debug|info|warn|error`.

## GET /api/config
The sanitized active `AppConfig` as JSON (`AppConfig::toJson()`).

## Calibration jobs
One job at a time; starting while busy returns `{"ok":false,"error":"busy"}`.

- `POST /api/calibrate/pair` body `{"reference":"front","target":"back",
  "min_samples":200,"max_seconds":60}` (last two optional).
- `POST /api/calibrate/body` body `{"seconds":15}` (optional, default 15).
- `POST /api/calibrate/playspace` body `{"hand":"right","seconds":20}`
  (optional). Responds 501 `{"ok":false,"error":"openvr client not built"}`
  when no PlayspaceFn is configured.
- `POST /api/calibrate/cancel` cancels the active job (`ok:true` always).

## GET /api/calibrate/status
```jsonc
{ "active": true, "kind": "pair",          // pair|body|playspace|"" when idle
  "phase": "collecting",                    // job-defined short string
  "progress": { "done": 120, "needed": 200 },
  "done": false,                            // a finished result is available
  "ok": false, "rmse_cm": 0.0, "message": "" }
```
After a job finishes, `active` goes false, `done` true, and `ok/rmse_cm/
message` describe the result until the next job starts. Successful jobs save
the CalibrationStore to disk; pair jobs also apply the extrinsic live. A
successful playspace job notes in `message` that the SteamVR bridge anchor
applies on next pipeline start.

## GET /api/doctor
Lightweight environment probe (no sensor capture):
```jsonc
{ "platform": "windows", "backends": { "kinect_v2": true, "kinect_v1": true },
  "openvr_client": true,
  "steamvr": { "openvrpaths_found": true, "driver_registered": false },
  "ports": { "dashboard": 8211, "wire": 24190 },
  "calibration": { "extrinsics": 2, "body_model_valid": false,
                    "world_anchor_set": false } }
```
`driver_registered`: whether the SteamVR `openvrpaths.vrpath` external_drivers
list contains a path ending in `marionette` (best effort; false on parse
failure).

## GET /api/projection
The live projection correction plus a validity probe of the most recent fused
frame.
```jsonc
{
  "flip_x": false, "flip_y": false, "flip_z": false, "swap_lr": false,
  "check": {
    "evaluated": true,          // false until the first body is fused
    "ok": true,                 // finite && in_bounds && bones_plausible && upright_ok
    "finite": true,             // no NaN/Inf in any tracked joint
    "in_bounds": true,          // all tracked joints within 4 m of the origin
    "bones_plausible": true,    // every connected bone within [0.02, 1.0] m
    "upright_ok": true,         // head sits above hips (catches a flipped-Y rig)
    "tracked_joints": 19, "non_finite_joints": 0, "out_of_bounds_joints": 0,
    "implausible_bones": 0, "worst_radius_m": 1.62, "worst_bone_error_m": 0.0,
    "head_above_hips_m": 0.65,  // clearly negative => vertical axis flipped
    "summary": "ok"             // "ok", "no body", or a problem list
  }
}
```

## POST /api/projection
Update the projection correction. All four keys are optional; an omitted key
keeps its current value. The change applies live on the next tick and is
persisted to the calibration store.
```jsonc
{ "flip_x": true, "flip_z": true }   // e.g. correct a backwards-facing rig
```
Responds with the same body as `GET /api/projection` plus `"ok"` and `"error"`.
`ok` is `true` whenever the correction was applied live; `error` is non-empty
only when persisting to the calibration file failed (the live fix still holds).

Common corrections: facing backwards -> `flip_x`+`flip_z`; mirrored left/right
-> `flip_x`+`swap_lr`; upside down -> `flip_y`.
