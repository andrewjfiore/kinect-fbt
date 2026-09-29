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
//
// Watchdog (cfg.watchdog): a monitor on the tick thread tracks per-node frame
// recency. A node silent past silentSeconds (and not of an excluded type) is
// recreated from its registry factory and restarted, with backoffSeconds
// between attempts and at most maxRestarts per node. Because of this, the
// NodeRegistry and EndpointRegistry passed to build() MUST outlive the
// Pipeline.

#include "mn/capture.hpp"
#include "mn/config.hpp"
#include "mn/endpoint.hpp"
#include "mn/fusion.hpp"
#include "mn/mapping.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mn {

// Raw frames: capture-thread context; must be cheap and thread-safe.
using RawFrameObserver = std::function<void(const NodeDescriptor&, const SkeletonFrame&)>;
// Fused output: tick-thread context, called only on ticks that fused a body.
using FusedFrameObserver =
    std::function<void(const SkeletonFrame& world, const std::vector<TrackerPose>& trackers)>;

class Pipeline {
public:
    struct Stats {
        uint64_t framesIn = 0;    // frames received from all nodes
        uint64_t ticks = 0;       // tick-loop iterations
        uint64_t fusedTicks = 0;  // ticks that produced a fused body
        uint64_t trackersOut = 0; // tracker poses pushed (sum over ticks)
        double lastFuseTimestamp = 0.0;
    };

    struct NodeStatus {
        std::string id;
        std::string type;
        bool running = false;
        uint64_t frames = 0;        // total frames received
        double lastFrameAge = -1.0; // seconds since last frame; -1 = never
        double fps = 0.0;           // ~2 s window estimate
        bool lastHasBody = false;
        uint32_t restarts = 0; // watchdog restarts so far
        std::string lastError;
    };

    // Builds nodes + endpoints. Returns nullptr with `error` set on failure.
    // `nodes`/`endpoints` registries must outlive the Pipeline (watchdog
    // recreates failed nodes through their factories).
    static std::unique_ptr<Pipeline> build(const AppConfig& cfg, const NodeRegistry& nodes,
                                           const EndpointRegistry& endpoints,
                                           const CalibrationStore& calib, std::string& error);

    ~Pipeline();

    bool start();              // starts nodes, endpoints, and the tick thread
    void stop();               // idempotent
    bool isRunning() const;
    Stats stats() const;

    // --- Introspection / dashboard hooks ---------------------------------
    std::vector<NodeStatus> nodeStatuses() const;
    SkeletonFrame latestFused() const;            // hasBody=false until first fuse
    std::vector<TrackerPose> latestTrackers() const;
    const AppConfig& appConfig() const;

    // Single observer slot each (replace; empty to clear).
    void setRawFrameObserver(RawFrameObserver cb);
    void setFusedFrameObserver(FusedFrameObserver cb);

    // Live-update a node's extrinsic in the fusion engine (after on-line
    // calibration). Returns false for unknown node ids.
    bool applyNodeExtrinsic(const std::string& nodeId, const Pose& extrinsic);

    // Projection correction (axis flips + left/right swap) applied to every
    // fused frame before it is mapped to trackers. Thread-safe; a change takes
    // effect on the next tick. Seeded from the CalibrationStore at build().
    void setProjectionCorrection(const ProjectionCorrection& c);
    ProjectionCorrection projectionCorrection() const;

    // Validity probe of the most recent fused frame (finiteness, bounds, bone
    // lengths, upright). evaluated=false until the first body is fused.
    ProjectionCheck latestProjectionCheck() const;

    FusionEngine& fusion();
    const std::vector<std::unique_ptr<ICaptureNode>>& captureNodes() const;

private:
    Pipeline();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mn
