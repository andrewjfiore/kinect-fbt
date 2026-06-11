#include <doctest/doctest.h>

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

struct EndpointCounters {
    std::atomic<uint64_t> pushes{0};
    std::atomic<uint64_t> trackersSeen{0};
    std::atomic<int> starts{0};
    std::atomic<int> stops{0};
};

class CountingEndpoint : public mn::IServiceEndpoint {
public:
    explicit CountingEndpoint(std::shared_ptr<EndpointCounters> c) : c_(std::move(c)) {}

    std::string name() const override { return "counting"; }
    bool start() override {
        c_->starts.fetch_add(1);
        return true;
    }
    void stop() override { c_->stops.fetch_add(1); }
    void push(const std::vector<mn::TrackerPose>& trackers, double /*timestamp*/) override {
        c_->pushes.fetch_add(1);
        c_->trackersSeen.fetch_add(static_cast<uint64_t>(trackers.size()));
    }

private:
    std::shared_ptr<EndpointCounters> c_;
};

mn::AppConfig makeConfig() {
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

    mn::EndpointConfigEntry ep;
    ep.type = "counting";
    ep.params = nlohmann::json::object();
    cfg.endpoints.push_back(ep);
    return cfg;
}

void registerCounting(mn::EndpointRegistry& reg, std::shared_ptr<EndpointCounters> counters) {
    reg.add("counting",
            [counters](const nlohmann::json&,
                       std::string&) -> std::unique_ptr<mn::IServiceEndpoint> {
                return std::make_unique<CountingEndpoint>(counters);
            });
}

} // namespace

TEST_CASE("Pipeline runs mock capture through fusion and mapping to the endpoint") {
    mn::NodeRegistry nodes;
    mn::mock::registerNodes(nodes);

    auto counters = std::make_shared<EndpointCounters>();
    mn::EndpointRegistry endpoints;
    registerCounting(endpoints, counters);

    mn::CalibrationStore calib; // empty: extrinsic comes from params["extrinsic"]
    std::string error;
    auto pipe = mn::Pipeline::build(makeConfig(), nodes, endpoints, calib, error);
    REQUIRE_MESSAGE(pipe != nullptr, error);
    CHECK(error.empty());
    CHECK(pipe->captureNodes().size() == 1u);
    CHECK_FALSE(pipe->isRunning());

    REQUIRE(pipe->start());
    CHECK(pipe->isRunning());
    CHECK(counters->starts.load() == 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    pipe->stop();
    CHECK_FALSE(pipe->isRunning());
    CHECK(counters->stops.load() == 1);

    const mn::Pipeline::Stats s = pipe->stats();
    CHECK(s.framesIn > 5u);
    CHECK(s.ticks > 10u);
    CHECK(s.fusedTicks > 10u);
    CHECK(s.fusedTicks <= s.ticks);
    CHECK(counters->pushes.load() > 10u);
    CHECK(counters->pushes.load() == s.fusedTicks);
    // Every push carried the configured trackers (3 default roles + Head).
    CHECK(s.trackersOut == counters->trackersSeen.load());
    CHECK(counters->trackersSeen.load() >= 4u * counters->pushes.load());
    CHECK(s.lastFuseTimestamp > 0.0);

    // stop() is idempotent: nothing restarts, nothing double-stops, stats freeze.
    const uint64_t ticksAfterStop = pipe->stats().ticks;
    const uint64_t pushesAfterStop = counters->pushes.load();
    pipe->stop();
    CHECK(counters->stops.load() == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(pipe->stats().ticks == ticksAfterStop);
    CHECK(counters->pushes.load() == pushesAfterStop);
    CHECK_FALSE(pipe->isRunning());
}

TEST_CASE("Pipeline build fails cleanly on unknown node type") {
    mn::NodeRegistry nodes; // empty: "mock" is unknown
    auto counters = std::make_shared<EndpointCounters>();
    mn::EndpointRegistry endpoints;
    registerCounting(endpoints, counters);

    mn::CalibrationStore calib;
    std::string error;
    auto pipe = mn::Pipeline::build(makeConfig(), nodes, endpoints, calib, error);
    CHECK(pipe == nullptr);
    CHECK_FALSE(error.empty());
    CHECK(error.find("front") != std::string::npos);
}

TEST_CASE("Pipeline destructor stops a running pipeline") {
    mn::NodeRegistry nodes;
    mn::mock::registerNodes(nodes);

    auto counters = std::make_shared<EndpointCounters>();
    mn::EndpointRegistry endpoints;
    registerCounting(endpoints, counters);

    mn::CalibrationStore calib;
    {
        std::string error;
        auto pipe = mn::Pipeline::build(makeConfig(), nodes, endpoints, calib, error);
        REQUIRE_MESSAGE(pipe != nullptr, error);
        REQUIRE(pipe->start());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // No explicit stop(): the destructor must shut everything down.
    }
    CHECK(counters->starts.load() == 1);
    CHECK(counters->stops.load() == 1);
    const uint64_t pushesAfterDestroy = counters->pushes.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(counters->pushes.load() == pushesAfterDestroy);
}
