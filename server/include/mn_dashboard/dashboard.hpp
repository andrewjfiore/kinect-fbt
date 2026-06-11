#pragma once
// Local web dashboard: live status, skeleton view, structured events,
// calibration wizard, and the onboarding tutorial. Serves a single embedded
// HTML app plus a JSON API (contract: docs/DASHBOARD_API.md). Binds loopback
// by default; bind 0.0.0.0 deliberately to expose on LAN/tailnet.
//
// Calibration jobs run on an internal worker thread, at most one at a time,
// using the pipeline's observer hooks (the dashboard owns the pipeline's raw
// and fused observer slots). Successful jobs persist the CalibrationStore to
// calibPath and (for pair calibration) apply the new extrinsic live via
// Pipeline::applyNodeExtrinsic.

#include "mn/calibration.hpp"
#include "mn/config.hpp"
#include "mn/pipeline.hpp"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace mn::dash {

using ProgressFn = std::function<void(size_t done, size_t needed, const std::string& phase)>;

// App-supplied playspace anchoring routine (requires the OpenVR client lib,
// which only the app links). Samples (fused wrist, SteamVR controller) pairs
// for `seconds`, honoring `cancel`; returns the world -> SteamVR fit or
// ok=false with `error` set.
using PlayspaceFn = std::function<RigidFit(const std::string& hand, double seconds,
                                           ProgressFn progress, std::atomic<bool>& cancel,
                                           std::string& error)>;

struct Options {
    std::string bind = "127.0.0.1";
    uint16_t port = 8211;
    // Serve index.html from this directory instead of the embedded copy
    // (frontend dev loop). Ignored when empty or missing.
    std::string webDirOverride;
    // Empty -> POST /api/calibrate/playspace responds 501 unavailable.
    PlayspaceFn playspace;
    // Compile-time capabilities of the host app, echoed in /api/status
    // ("kinect_v2", "kinect_v1", "openvr_client", ...).
    std::map<std::string, bool> capabilities;
};

class DashboardServer {
public:
    // `pipeline` and `store` must outlive the server. `calibPath` is where
    // the store is persisted after successful calibration jobs.
    DashboardServer(Pipeline& pipeline, CalibrationStore& store, std::string calibPath,
                    Options opt = {});
    ~DashboardServer();

    bool start(); // binds and spawns the server thread; false -> lastError()
    void stop();  // idempotent; cancels any active calibration job
    bool isRunning() const;
    std::string url() const; // e.g. "http://127.0.0.1:8211"
    std::string lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mn::dash
