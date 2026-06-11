# Roadmap

## Phase 1 - this repo, current scope
- [x] Core contracts: joint schema, capture/endpoint plugins, pipeline
- [x] Mock + replay nodes (zero-hardware dev loop)
- [x] Weighted fusion + One-Euro + bone constraint; Kabsch/Horn calibration
- [x] VRChat OSC Trackers endpoint (Quest standalone path)
- [x] OpenVR driver + UDP bridge (PCVR path: ALVR / Virtual Desktop / Link)
- [x] Kinect v2 + v1 backends (MS SDKs, Windows)
- [x] Tests + Windows/Linux CI

## Phase 2 - hardware bring-up & quality
- [x] Web dashboard: live status, skeleton view, events, calibration wizards,
      onboarding tutorial (pulled forward from the phase-4 GUI item)
- [x] `marionette doctor`: backend/sensor/config/calibration/ports/SteamVR probe
      (`--json` for machine use)
- [x] Node watchdog: factory-recreate + restart with backoff for silent sensors
- [x] Cross-node outlier rejection in fusion (consensus-of-others downweighting)
- [x] Hardware smoke test (`ctest -L hardware`, skips cleanly without sensors)
- [ ] On-hardware validation: 1x Kinect v2, 3-point FBT in VRChat via ALVR
- [ ] Pair calibration UX polish (live sample-coverage feedback, RMSE gate)
- [ ] Playspace anchor validation against OpenVR-SpaceCalibrator
- [ ] Body model auto-calibration on session start
- [ ] Skeleton-flip guard for single-sensor rigs (HMD yaw hint via OSC query / OpenVR)

## Phase 3 - scale out
- [ ] Remote capture nodes (extra v2s on other PCs, LAN frame stream)
- [ ] EKF fusion tier (root pose + joint angles, per-sensor covariances)
- [ ] libfreenect2 markerless backend (RTMPose/BlazePose + depth lift, ONNX Runtime)
- [ ] Multi-v2-per-box USB topology guide + bring-up tooling

## Phase 4 - polish
- [x] GUI (status, placement viz, calibration wizard) - shipped early as the
      phase-2 web dashboard
- [ ] Installer + SteamVR driver auto-registration
- [ ] Quest 2 QoS presets (lower bitrate budgets, stronger prediction)
