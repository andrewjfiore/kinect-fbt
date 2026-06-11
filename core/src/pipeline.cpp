#include "mn/pipeline.hpp"

#include "mn/clock.hpp"
#include "mn/log.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

namespace mn {

namespace {
constexpr double kFpsWindowSeconds = 2.0;      // sliding window for the fps estimate
constexpr double kWatchdogPeriodSeconds = 1.0; // watchdog check cadence on the tick thread
} // namespace

struct Pipeline::Impl {
    AppConfig cfg;
    const NodeRegistry* nodeRegistry = nullptr; // must outlive the Pipeline (see header)
    FusionEngine fusionEngine;
    TrackerMapper mapper;

    // Per-node runtime state. Stable-addressed (unique_ptr) so the shared
    // frame callback can hold pointers while the watchdog swaps nodes.
    struct NodeState {
        NodeConfigEntry entry; // original config entry, used to recreate the node
        std::atomic<bool> started{false}; // start() succeeded at least once
        std::atomic<bool> givenUp{false}; // maxRestarts exhausted; left stopped
        std::atomic<uint64_t> frames{0};
        std::atomic<bool> lastHasBody{false};
        std::atomic<uint32_t> restarts{0};
        std::atomic<double> lastFrameTime{-1.0};      // actual frames only; -1 = never
        std::atomic<double> lastStartTime{-1.0};      // (re)start time; silence baseline
        std::atomic<double> nextRestartAllowed{0.0};  // backoff gate
        mutable std::mutex mtx;                       // guards lastError + frameTimes
        std::string lastError;
        std::deque<double> frameTimes; // recent frame arrival times (fps window)
    };

    std::vector<std::unique_ptr<ICaptureNode>> nodes; // parallel to `states`
    std::vector<std::unique_ptr<NodeState>> states;
    std::map<std::string, NodeState*> stateById; // immutable after build()
    // Guards the node slots themselves (the watchdog swaps unique_ptrs on the
    // tick thread while stop()/nodeStatuses() run on other threads).
    mutable std::mutex nodesMutex;

    std::vector<std::unique_ptr<IServiceEndpoint>> endpoints;

    // The wrapped frame callback handed to every node; reused verbatim by the
    // watchdog when it restarts a recreated node.
    FrameCallback frameCb;

    // Observer slots: stored behind shared_ptr and swapped under a mutex so
    // replacing one mid-run is safe against in-flight invocations.
    std::mutex obsMutex;
    std::shared_ptr<RawFrameObserver> rawObs;
    std::shared_ptr<FusedFrameObserver> fusedObs;

    mutable std::mutex dataMutex;
    SkeletonFrame latestFused;               // default (hasBody=false) until first fuse
    std::vector<TrackerPose> latestTrackers; // empty until first fuse

    std::thread tickThread;
    std::atomic<bool> running{false};
    std::atomic<bool> stopRequested{false};

    std::atomic<uint64_t> framesIn{0};
    std::atomic<uint64_t> ticks{0};
    std::atomic<uint64_t> fusedTicks{0};
    std::atomic<uint64_t> trackersOut{0};
    std::atomic<double> lastFuseTimestamp{0.0};

    explicit Impl(const AppConfig& c) : cfg(c), fusionEngine(c.fusion), mapper(c.mapping) {}

    void setLastError(NodeState& st, const std::string& err) {
        if (err.empty())
            return;
        std::lock_guard<std::mutex> lk(st.mtx);
        st.lastError = err;
    }

    // Capture-thread bookkeeping for one delivered frame.
    void noteFrame(NodeState& st, const SkeletonFrame& frame) {
        const double now = nowSeconds();
        st.frames.fetch_add(1, std::memory_order_relaxed);
        st.lastHasBody.store(frame.hasBody, std::memory_order_relaxed);
        st.lastFrameTime.store(now, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(st.mtx);
        st.frameTimes.push_back(now);
        const double cutoff = now - kFpsWindowSeconds;
        while (!st.frameTimes.empty() && st.frameTimes.front() < cutoff)
            st.frameTimes.pop_front();
    }

    std::shared_ptr<RawFrameObserver> rawObserver() {
        std::lock_guard<std::mutex> lk(obsMutex);
        return rawObs;
    }

    std::shared_ptr<FusedFrameObserver> fusedObserver() {
        std::lock_guard<std::mutex> lk(obsMutex);
        return fusedObs;
    }

    // Watchdog: runs on the tick thread at ~1 Hz. A started, non-excluded node
    // that is not running or has been silent past silentSeconds is recreated
    // from its registry factory and restarted with the same wrapped callback.
    // "Silent" includes never having delivered a frame since the last
    // (re)start: the silence baseline is max(lastFrameTime, lastStartTime).
    void runWatchdog(double now) {
        const WatchdogConfig& wd = cfg.watchdog;
        std::lock_guard<std::mutex> lk(nodesMutex);
        for (size_t i = 0; i < nodes.size(); ++i) {
            NodeState& st = *states[i];
            if (!st.started.load() || st.givenUp.load())
                continue;
            if (std::find(wd.excludeTypes.begin(), wd.excludeTypes.end(), st.entry.type) !=
                wd.excludeTypes.end())
                continue;

            ICaptureNode& node = *nodes[i];
            const double lastSeen =
                std::max(st.lastFrameTime.load(std::memory_order_relaxed),
                         st.lastStartTime.load(std::memory_order_relaxed));
            const bool silent = !node.isRunning() || (now - lastSeen) > wd.silentSeconds;
            if (!silent)
                continue;
            if (now < st.nextRestartAllowed.load(std::memory_order_relaxed))
                continue; // backoff between attempts

            setLastError(st, node.lastError()); // node stopped/stalled unexpectedly

            if (st.restarts.load() >= wd.maxRestarts) {
                log::error("pipeline: watchdog: node \"", st.entry.id, "\" still silent after ",
                           st.restarts.load(), " restarts (max ", wd.maxRestarts,
                           "); leaving it stopped");
                node.stop();
                st.givenUp.store(true);
                continue;
            }

            node.stop();
            st.restarts.fetch_add(1);
            st.nextRestartAllowed.store(now + wd.backoffSeconds, std::memory_order_relaxed);

            std::string err;
            auto fresh = nodeRegistry->create(st.entry.type, st.entry.id, st.entry.params, err);
            if (!fresh) {
                setLastError(st, err.empty() ? std::string("factory returned null") : err);
                log::warn("pipeline: watchdog: node \"", st.entry.id,
                          "\" recreate failed (attempt ", st.restarts.load(), "/", wd.maxRestarts,
                          "): ", err);
                continue; // keep the old (stopped) node in the slot
            }
            if (!fresh->start(frameCb)) {
                setLastError(st, fresh->lastError());
                log::warn("pipeline: watchdog: node \"", st.entry.id,
                          "\" restart failed (attempt ", st.restarts.load(), "/", wd.maxRestarts,
                          "): ", fresh->lastError());
                nodes[i] = std::move(fresh);
                continue;
            }
            st.lastStartTime.store(nowSeconds(), std::memory_order_relaxed);
            nodes[i] = std::move(fresh); // old node already stopped; safe to destroy
            log::warn("pipeline: watchdog restarted node \"", st.entry.id, "\" (restart ",
                      st.restarts.load(), "/", wd.maxRestarts, ")");
        }
    }

    void tickLoop() {
        using Clock = std::chrono::steady_clock;
        const double hz = (cfg.tickHz > 0.0) ? cfg.tickHz : 90.0;
        const auto period =
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / hz));
        auto next = Clock::now() + period;
        double lastWatchdog = nowSeconds();
        while (!stopRequested.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_until(next);
            if (stopRequested.load(std::memory_order_relaxed))
                break;
            ticks.fetch_add(1, std::memory_order_relaxed);

            const double now = nowSeconds();
            SkeletonFrame world;
            if (fusionEngine.fuse(now, world)) {
                const std::vector<TrackerPose> trackers = mapper.map(world, now);
                {
                    std::lock_guard<std::mutex> lk(dataMutex);
                    latestFused = world;
                    latestTrackers = trackers;
                }
                if (auto obs = fusedObserver())
                    (*obs)(world, trackers);
                for (const auto& ep : endpoints)
                    ep->push(trackers, now);
                fusedTicks.fetch_add(1, std::memory_order_relaxed);
                trackersOut.fetch_add(static_cast<uint64_t>(trackers.size()),
                                      std::memory_order_relaxed);
                lastFuseTimestamp.store(now, std::memory_order_relaxed);
            }

            if (cfg.watchdog.enable && (now - lastWatchdog) >= kWatchdogPeriodSeconds) {
                lastWatchdog = now;
                runWatchdog(now);
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
    im.nodeRegistry = &nodes;

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

        auto state = std::make_unique<Impl::NodeState>();
        state->entry = entry;
        im.stateById[entry.id] = state.get();
        im.states.push_back(std::move(state));
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
    im.frameCb = [imp](const NodeDescriptor& desc, const SkeletonFrame& frame) {
        imp->framesIn.fetch_add(1, std::memory_order_relaxed);
        imp->fusionEngine.submit(desc.id, frame);
        auto it = imp->stateById.find(desc.id);
        if (it != imp->stateById.end())
            imp->noteFrame(*it->second, frame);
        if (auto obs = imp->rawObserver())
            (*obs)(desc, frame);
    };

    size_t ndStarted = 0;
    for (; ndStarted < im.nodes.size(); ++ndStarted) {
        ICaptureNode& node = *im.nodes[ndStarted];
        Impl::NodeState& st = *im.states[ndStarted];
        if (!node.start(im.frameCb)) {
            im.setLastError(st, node.lastError());
            log::error("pipeline: node \"", node.descriptor().id,
                       "\" failed to start: ", node.lastError());
            for (size_t k = 0; k < ndStarted; ++k)
                im.nodes[k]->stop();
            for (auto& ep : im.endpoints)
                ep->stop();
            return false;
        }
        st.started.store(true);
        // Silence baseline: a node that never delivers a frame after this
        // (re)start counts as silent once silentSeconds elapse.
        st.lastStartTime.store(nowSeconds(), std::memory_order_relaxed);
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
    // Join the tick thread first: it hosts the watchdog, which may otherwise
    // swap/restart nodes underneath us. Once it is gone, stop the nodes (no
    // more frames) and finally the endpoints (no more pushes by then).
    if (im.tickThread.joinable())
        im.tickThread.join();
    if (!wasRunning)
        return;
    {
        std::lock_guard<std::mutex> lk(im.nodesMutex);
        for (auto& node : im.nodes)
            node->stop();
    }
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

std::vector<Pipeline::NodeStatus> Pipeline::nodeStatuses() const {
    const Impl& im = *impl_;
    std::vector<NodeStatus> out;
    const double now = nowSeconds();
    std::lock_guard<std::mutex> lk(im.nodesMutex);
    out.reserve(im.nodes.size());
    for (size_t i = 0; i < im.nodes.size(); ++i) {
        const Impl::NodeState& st = *im.states[i];
        NodeStatus s;
        s.id = st.entry.id;
        s.type = st.entry.type;
        s.running = im.nodes[i]->isRunning();
        s.frames = st.frames.load(std::memory_order_relaxed);
        const double lastFrame = st.lastFrameTime.load(std::memory_order_relaxed);
        // `now` was sampled at entry; a frame can land in between, so clamp.
        s.lastFrameAge = (lastFrame < 0.0) ? -1.0 : std::max(0.0, now - lastFrame);
        s.lastHasBody = st.lastHasBody.load(std::memory_order_relaxed);
        s.restarts = st.restarts.load();
        {
            std::lock_guard<std::mutex> lk2(st.mtx);
            s.lastError = st.lastError;
            const double cutoff = now - kFpsWindowSeconds;
            size_t n = 0;
            for (double t : st.frameTimes) {
                if (t >= cutoff)
                    ++n;
            }
            double denom = kFpsWindowSeconds;
            const double startedAt = st.lastStartTime.load(std::memory_order_relaxed);
            if (startedAt >= 0.0 && (now - startedAt) < kFpsWindowSeconds)
                denom = std::max(now - startedAt, 1e-3);
            s.fps = static_cast<double>(n) / denom;
        }
        out.push_back(std::move(s));
    }
    return out;
}

SkeletonFrame Pipeline::latestFused() const {
    std::lock_guard<std::mutex> lk(impl_->dataMutex);
    return impl_->latestFused;
}

std::vector<TrackerPose> Pipeline::latestTrackers() const {
    std::lock_guard<std::mutex> lk(impl_->dataMutex);
    return impl_->latestTrackers;
}

const AppConfig& Pipeline::appConfig() const { return impl_->cfg; }

void Pipeline::setRawFrameObserver(RawFrameObserver cb) {
    std::shared_ptr<RawFrameObserver> p;
    if (cb)
        p = std::make_shared<RawFrameObserver>(std::move(cb));
    std::lock_guard<std::mutex> lk(impl_->obsMutex);
    impl_->rawObs = std::move(p);
}

void Pipeline::setFusedFrameObserver(FusedFrameObserver cb) {
    std::shared_ptr<FusedFrameObserver> p;
    if (cb)
        p = std::make_shared<FusedFrameObserver>(std::move(cb));
    std::lock_guard<std::mutex> lk(impl_->obsMutex);
    impl_->fusedObs = std::move(p);
}

bool Pipeline::applyNodeExtrinsic(const std::string& nodeId, const Pose& extrinsic) {
    Impl& im = *impl_;
    if (im.stateById.find(nodeId) == im.stateById.end())
        return false;
    im.fusionEngine.setNode(nodeId, extrinsic);
    return true;
}

FusionEngine& Pipeline::fusion() { return impl_->fusionEngine; }

const std::vector<std::unique_ptr<ICaptureNode>>& Pipeline::captureNodes() const {
    return impl_->nodes;
}

} // namespace mn
