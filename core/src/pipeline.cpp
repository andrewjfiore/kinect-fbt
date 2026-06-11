#include "mn/pipeline.hpp"

#include "mn/clock.hpp"
#include "mn/log.hpp"

#include <chrono>
#include <set>
#include <thread>
#include <utility>

namespace mn {

struct Pipeline::Impl {
    AppConfig cfg;
    FusionEngine fusionEngine;
    TrackerMapper mapper;
    std::vector<std::unique_ptr<ICaptureNode>> nodes;
    std::vector<std::unique_ptr<IServiceEndpoint>> endpoints;

    std::thread tickThread;
    std::atomic<bool> running{false};
    std::atomic<bool> stopRequested{false};

    std::atomic<uint64_t> framesIn{0};
    std::atomic<uint64_t> ticks{0};
    std::atomic<uint64_t> fusedTicks{0};
    std::atomic<uint64_t> trackersOut{0};
    std::atomic<double> lastFuseTimestamp{0.0};

    explicit Impl(const AppConfig& c) : cfg(c), fusionEngine(c.fusion), mapper(c.mapping) {}

    void tickLoop() {
        using Clock = std::chrono::steady_clock;
        const double hz = (cfg.tickHz > 0.0) ? cfg.tickHz : 90.0;
        const auto period =
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / hz));
        auto next = Clock::now() + period;
        while (!stopRequested.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_until(next);
            if (stopRequested.load(std::memory_order_relaxed))
                break;
            ticks.fetch_add(1, std::memory_order_relaxed);

            const double now = nowSeconds();
            SkeletonFrame world;
            if (fusionEngine.fuse(now, world)) {
                const std::vector<TrackerPose> trackers = mapper.map(world, now);
                for (const auto& ep : endpoints)
                    ep->push(trackers, now);
                fusedTicks.fetch_add(1, std::memory_order_relaxed);
                trackersOut.fetch_add(static_cast<uint64_t>(trackers.size()),
                                      std::memory_order_relaxed);
                lastFuseTimestamp.store(now, std::memory_order_relaxed);
            }

            next += period;
            const auto wall = Clock::now();
            if (next < wall)
                next = wall + period; // fell behind; drop the missed ticks and resync
        }
    }
};

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() { stop(); }

std::unique_ptr<Pipeline> Pipeline::build(const AppConfig& cfg, const NodeRegistry& nodes,
                                          const EndpointRegistry& endpoints,
                                          const CalibrationStore& calib, std::string& error) {
    error.clear();
    std::unique_ptr<Pipeline> p(new Pipeline());
    p->impl_ = std::make_unique<Impl>(cfg);
    Impl& im = *p->impl_;

    std::set<std::string> ids;
    for (const auto& entry : cfg.nodes) {
        if (!ids.insert(entry.id).second) {
            error = "pipeline: duplicate node id \"" + entry.id + "\"";
            return nullptr;
        }

        // Extrinsic resolution: CalibrationStore > params["extrinsic"] > identity.
        Pose extrinsic = Pose::identity();
        if (const auto calibrated = calib.nodeExtrinsic(entry.id)) {
            extrinsic = *calibrated;
        } else if (entry.params.is_object() && entry.params.contains("extrinsic")) {
            try {
                extrinsic = entry.params.at("extrinsic").get<Pose>();
            } catch (const std::exception& e) {
                error = "pipeline: node \"" + entry.id + "\": bad params.extrinsic: " + e.what();
                return nullptr;
            }
        } else {
            log::warn("pipeline: node \"", entry.id,
                      "\" has no extrinsic (calibration or params); using identity");
        }

        std::string nodeError;
        auto node = nodes.create(entry.type, entry.id, entry.params, nodeError);
        if (!node) {
            error = "pipeline: node \"" + entry.id + "\" (type \"" + entry.type + "\"): " +
                    (nodeError.empty() ? std::string("factory returned null") : nodeError);
            return nullptr;
        }
        im.fusionEngine.setNode(entry.id, extrinsic);
        im.nodes.push_back(std::move(node));
    }

    if (calib.bodyModel().valid)
        im.fusionEngine.setBodyModel(calib.bodyModel());

    for (const auto& entry : cfg.endpoints) {
        nlohmann::json params = entry.params.is_null() ? nlohmann::json::object() : entry.params;
        // The OpenVR bridge maps Marionette world -> SteamVR playspace; hand it
        // the calibrated anchor (identity until calibrated). OSC stays in world.
        if (entry.type == "openvr")
            params["world_anchor"] = calib.worldAnchor().value_or(Pose::identity());

        std::string epError;
        auto ep = endpoints.create(entry.type, params, epError);
        if (!ep) {
            error = "pipeline: endpoint type \"" + entry.type + "\": " +
                    (epError.empty() ? std::string("factory returned null") : epError);
            return nullptr;
        }
        im.endpoints.push_back(std::move(ep));
    }

    return p;
}

bool Pipeline::start() {
    Impl& im = *impl_;
    if (im.running.load())
        return true;

    // Endpoints first: they must be ready before the first tick can push.
    size_t epStarted = 0;
    for (; epStarted < im.endpoints.size(); ++epStarted) {
        IServiceEndpoint& ep = *im.endpoints[epStarted];
        if (!ep.start()) {
            log::error("pipeline: endpoint \"", ep.name(), "\" failed to start: ", ep.lastError());
            for (size_t k = 0; k < epStarted; ++k)
                im.endpoints[k]->stop();
            return false;
        }
    }

    Impl* imp = impl_.get();
    FrameCallback cb = [imp](const NodeDescriptor& desc, const SkeletonFrame& frame) {
        imp->framesIn.fetch_add(1, std::memory_order_relaxed);
        imp->fusionEngine.submit(desc.id, frame);
    };

    size_t ndStarted = 0;
    for (; ndStarted < im.nodes.size(); ++ndStarted) {
        ICaptureNode& node = *im.nodes[ndStarted];
        if (!node.start(cb)) {
            log::error("pipeline: node \"", node.descriptor().id,
                       "\" failed to start: ", node.lastError());
            for (size_t k = 0; k < ndStarted; ++k)
                im.nodes[k]->stop();
            for (auto& ep : im.endpoints)
                ep->stop();
            return false;
        }
    }

    im.stopRequested.store(false);
    im.running.store(true);
    im.tickThread = std::thread([imp] { imp->tickLoop(); });
    return true;
}

void Pipeline::stop() {
    if (!impl_)
        return;
    Impl& im = *impl_;
    im.stopRequested.store(true);
    const bool wasRunning = im.running.exchange(false);
    if (!wasRunning) {
        if (im.tickThread.joinable())
            im.tickThread.join();
        return;
    }
    // Nodes first (no more frames), then the tick thread (no more pushes),
    // then the endpoints.
    for (auto& node : im.nodes)
        node->stop();
    if (im.tickThread.joinable())
        im.tickThread.join();
    for (auto& ep : im.endpoints)
        ep->stop();
}

bool Pipeline::isRunning() const { return impl_ && impl_->running.load(); }

Pipeline::Stats Pipeline::stats() const {
    const Impl& im = *impl_;
    Stats s;
    s.framesIn = im.framesIn.load(std::memory_order_relaxed);
    s.ticks = im.ticks.load(std::memory_order_relaxed);
    s.fusedTicks = im.fusedTicks.load(std::memory_order_relaxed);
    s.trackersOut = im.trackersOut.load(std::memory_order_relaxed);
    s.lastFuseTimestamp = im.lastFuseTimestamp.load(std::memory_order_relaxed);
    return s;
}

FusionEngine& Pipeline::fusion() { return impl_->fusionEngine; }

const std::vector<std::unique_ptr<ICaptureNode>>& Pipeline::captureNodes() const {
    return impl_->nodes;
}

} // namespace mn
