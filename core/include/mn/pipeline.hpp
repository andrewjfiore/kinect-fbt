#pragma once
// Wires everything together: capture nodes -> FusionEngine -> TrackerMapper
// -> endpoints, ticking at cfg.tickHz on a dedicated thread.
//
// Extrinsic resolution order per node: CalibrationStore entry, else node
// params["extrinsic"], else identity (with a warning).
//
// The world anchor (CalibrationStore) is injected into "openvr" endpoint
// params as params["world_anchor"] so the bridge can map Marionette world ->
// SteamVR playspace; OSC output stays in Marionette world (VRChat aligns via
// the Head reference).

#include "mn/capture.hpp"
#include "mn/config.hpp"
#include "mn/endpoint.hpp"
#include "mn/fusion.hpp"
#include "mn/mapping.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace mn {

class Pipeline {
public:
    struct Stats {
        uint64_t framesIn = 0;    // frames received from all nodes
        uint64_t ticks = 0;       // tick-loop iterations
        uint64_t fusedTicks = 0;  // ticks that produced a fused body
        uint64_t trackersOut = 0; // tracker poses pushed (sum over ticks)
        double lastFuseTimestamp = 0.0;
    };

    // Builds nodes + endpoints. Returns nullptr with `error` set on failure.
    static std::unique_ptr<Pipeline> build(const AppConfig& cfg, const NodeRegistry& nodes,
                                           const EndpointRegistry& endpoints,
                                           const CalibrationStore& calib, std::string& error);

    ~Pipeline();

    bool start();              // starts nodes, endpoints, and the tick thread
    void stop();               // idempotent
    bool isRunning() const;
    Stats stats() const;

    FusionEngine& fusion();
    const std::vector<std::unique_ptr<ICaptureNode>>& captureNodes() const;

private:
    Pipeline();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mn
