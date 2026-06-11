#include <doctest/doctest.h>

#include "mn/clock.hpp"
#include "mn/pipeline.hpp"
#include "mn_mock/mock.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<int> g_flakyCreations{0};

// Emits frames at ~30 Hz for the first 0.2 s after each (re)creation, then
// goes silent while still reporting isRunning() == true (exercising the
// frame-recency half of the watchdog, not the isRunning() half). The static
// creation counter lets tests observe watchdog recreations deterministically.
class FlakyNode final : public mn::ICaptureNode {
public:
    explicit FlakyNode(std::string id) : desc_{std::move(id), "flaky"} {
        g_flakyCreations.fetch_add(1);
    }
    ~FlakyNode() override { join(); }

    const mn::NodeDescriptor& descriptor() const override { return desc_; }

    bool start(mn::FrameCallback cb) override {
        if (running_.load())
            return true;
        stop_.store(false);
        running_.store(true);
        thread_ = std::thread([this, cb = std::move(cb)] {
            const double t0 = mn::nowSeconds();
            while (!stop_.load()) {
                const double now = mn::nowSeconds();
                if (now - t0 < 0.2) {
                    mn::SkeletonFrame f = mn::mock::groundTruth(now);
                    f.timestamp = now;
                    cb(desc_, f);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        });
        return true;
    }

    void stop() override { join(); }
    bool isRunning() const override { return running_.load(); }

private:
    void join() {
        stop_.store(true);
        if (thread_.joinable())
            thread_.join();
        running_.store(false);
    }

    mn::NodeDescriptor desc_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
};

void registerFlaky(mn::NodeRegistry& reg) {
    reg.add("flaky",
            [](const std::string& id, const nlohmann::json&,
               std::string&) -> std::unique_ptr<mn::ICaptureNode> {
                return std::make_unique<FlakyNode>(id);
            });
}

mn::AppConfig flakyConfig() {
    mn::AppConfig cfg;
    cfg.tickHz = 60.0;

    mn::NodeConfigEntry node;
    node.id = "cam";
    node.type = "flaky";
    node.params = nlohmann::json{
        {"extrinsic", nlohmann::json{{"pos", {0.0, 0.0, 0.0}}, {"rot", {1.0, 0.0, 0.0, 0.0}}}}};
    cfg.nodes.push_back(node);

    cfg.watchdog.enable = true;
    cfg.watchdog.silentSeconds = 0.4;
    cfg.watchdog.backoffSeconds = 0.1;
    cfg.watchdog.maxRestarts = 2;
    cfg.watchdog.excludeTypes.clear();
    return cfg;
}

mn::AppConfig mockConfig() {
    mn::AppConfig cfg;
    cfg.tickHz = 60.0;

    mn::NodeConfigEntry node;
    node.id = "front";
    node.type = "mock";
    node.params = nlohmann::json{
        {"pattern", "walk_in_place"},
        {"rate_hz", 30.0},
        {"noise_m", 0.0},
        {"seed", 1},
        {"confidence", 0.9},
        {"extrinsic", nlohmann::json{{"pos", {0.0, 0.0, 0.0}}, {"rot", {1.0, 0.0, 0.0, 0.0}}}},
    };
    cfg.nodes.push_back(node);
    return cfg;
}

} // namespace

TEST_CASE("Watchdog restarts a silent node and respects maxRestarts") {
    mn::NodeRegistry nodes;
    registerFlaky(nodes);
    mn::EndpointRegistry endpoints;
    mn::CalibrationStore calib;

    std::string error;
    auto pipe = mn::Pipeline::build(flakyConfig(), nodes, endpoints, calib, error);
    REQUIRE_MESSAGE(pipe != nullptr, error);
    const int creationsAfterBuild = g_flakyCreations.load();
    REQUIRE(pipe->start());

    // The node emits for 0.2 s then stalls; with ~1 Hz watchdog checks the two
    // restarts land around t=1 s and t=2 s. Poll with headroom for slow CI.
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(4000);
    uint32_t restarts = 0;
    while (Clock::now() < deadline) {
        const auto statuses = pipe->nodeStatuses();
        REQUIRE(statuses.size() == 1u);
        restarts = statuses[0].restarts;
        if (restarts >= 2u)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(restarts == 2u);
    CHECK(g_flakyCreations.load() == creationsAfterBuild + 2); // observed recreations
    CHECK(pipe->isRunning());

    // One more watchdog cycle: maxRestarts must not be exceeded (the node is
    // declared dead and left stopped; the pipeline itself keeps running).
    std::this_thread::sleep_for(std::chrono::milliseconds(1300));
    const auto statuses = pipe->nodeStatuses();
    REQUIRE(statuses.size() == 1u);
    CHECK(statuses[0].restarts == 2u);
    CHECK(statuses[0].frames > 0u);
    CHECK(statuses[0].id == "cam");
    CHECK(statuses[0].type == "flaky");
    CHECK(g_flakyCreations.load() == creationsAfterBuild + 2);
    CHECK(pipe->isRunning());

    pipe->stop();
    CHECK_FALSE(pipe->isRunning());
    pipe->stop(); // idempotent
    CHECK_FALSE(pipe->isRunning());
}

TEST_CASE("Watchdog leaves excluded node types alone") {
    mn::NodeRegistry nodes;
    registerFlaky(nodes);
    mn::EndpointRegistry endpoints;
    mn::CalibrationStore calib;

    auto cfg = flakyConfig();
    cfg.watchdog.excludeTypes = {"flaky"};

    std::string error;
    auto pipe = mn::Pipeline::build(cfg, nodes, endpoints, calib, error);
    REQUIRE_MESSAGE(pipe != nullptr, error);
    const int creationsAfterBuild = g_flakyCreations.load();
    REQUIRE(pipe->start());

    // Long enough for at least one watchdog check well past silentSeconds.
    std::this_thread::sleep_for(std::chrono::milliseconds(1800));

    const auto statuses = pipe->nodeStatuses();
    REQUIRE(statuses.size() == 1u);
    CHECK(statuses[0].restarts == 0u);
    CHECK(statuses[0].frames > 0u);
    CHECK(g_flakyCreations.load() == creationsAfterBuild); // never recreated

    pipe->stop();
    CHECK_FALSE(pipe->isRunning());
}

TEST_CASE("Observers fire and latest snapshots populate") {
    mn::NodeRegistry nodes;
    mn::mock::registerNodes(nodes);
    mn::EndpointRegistry endpoints;
    mn::CalibrationStore calib;

    std::string error;
    auto pipe = mn::Pipeline::build(mockConfig(), nodes, endpoints, calib, error);
    REQUIRE_MESSAGE(pipe != nullptr, error);

    // Defaults before any fuse.
    CHECK_FALSE(pipe->latestFused().hasBody);
    CHECK(pipe->latestTrackers().empty());
    REQUIRE(pipe->appConfig().nodes.size() == 1u);
    CHECK(pipe->appConfig().nodes[0].id == "front");

    std::atomic<uint64_t> raw{0};
    std::atomic<uint64_t> fused{0};
    std::atomic<bool> rawPayloadOk{true};
    std::atomic<bool> fusedPayloadOk{true};
    pipe->setRawFrameObserver([&](const mn::NodeDescriptor& d, const mn::SkeletonFrame& f) {
        raw.fetch_add(1);
        if (d.id != "front" || d.type != "mock" || !f.hasBody)
            rawPayloadOk.store(false);
    });
    pipe->setFusedFrameObserver(
        [&](const mn::SkeletonFrame& world, const std::vector<mn::TrackerPose>& trackers) {
            fused.fetch_add(1);
            if (!world.hasBody || trackers.empty())
                fusedPayloadOk.store(false);
        });

    REQUIRE(pipe->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(900));

    CHECK(raw.load() > 5u);
    CHECK(fused.load() > 5u);
    CHECK(rawPayloadOk.load());
    CHECK(fusedPayloadOk.load());
    CHECK(pipe->latestFused().hasBody);
    CHECK_FALSE(pipe->latestTrackers().empty());

    const auto statuses = pipe->nodeStatuses();
    REQUIRE(statuses.size() == 1u);
    CHECK(statuses[0].running);
    CHECK(statuses[0].frames > 0u);
    CHECK(statuses[0].lastHasBody);
    CHECK(statuses[0].lastFrameAge >= 0.0);
    CHECK(statuses[0].fps > 5.0);

    // applyNodeExtrinsic: known id accepted, unknown id refused.
    CHECK(pipe->applyNodeExtrinsic("front", mn::Pose::identity()));
    CHECK_FALSE(pipe->applyNodeExtrinsic("nope", mn::Pose::identity()));

    // Replacing/clearing an observer mid-run is safe and actually detaches.
    pipe->setRawFrameObserver({});
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    const uint64_t rawAfterClear = raw.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(raw.load() == rawAfterClear);

    pipe->stop();
    CHECK_FALSE(pipe->isRunning());
}
