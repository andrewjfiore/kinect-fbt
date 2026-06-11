// Full-stack end-to-end test, no hardware: two mock capture nodes -> fusion ->
// mapping -> OSC endpoint over real UDP loopback -> receiver socket -> OSC
// parser, with assertions on addresses, finiteness, and converted waist height.

#include <doctest/doctest.h>

#include "mn/config.hpp"
#include "mn/net.hpp"
#include "mn/pipeline.hpp"
#include "mn_mock/mock.hpp"
#include "mn_osc/osc.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint16_t kPort = 9123;

nlohmann::json e2eConfigJson() {
    return nlohmann::json::parse(R"json({
        "tick_hz": 60,
        "calibration_file": "calibration.json",
        "nodes": [
            {
                "id": "front",
                "type": "mock",
                "params": {
                    "pattern": "walk_in_place",
                    "rate_hz": 30,
                    "noise_m": 0.004,
                    "seed": 1,
                    "confidence": 0.9,
                    "view_pose": { "pos": [0.0, 0.9, -2.0], "rot": [1, 0, 0, 0] },
                    "extrinsic": { "pos": [0.0, 0.9, -2.0], "rot": [1, 0, 0, 0] }
                }
            },
            {
                "id": "back",
                "type": "mock",
                "params": {
                    "pattern": "walk_in_place",
                    "rate_hz": 30,
                    "noise_m": 0.004,
                    "seed": 2,
                    "confidence": 0.9,
                    "view_pose": { "pos": [0.0, 0.9, 2.0], "rot": [0, 0, 1, 0] },
                    "extrinsic": { "pos": [0.0, 0.9, 2.0], "rot": [0, 0, 1, 0] }
                }
            }
        ],
        "fusion": {
            "stale_seconds": 0.15,
            "inferred_weight": 0.25,
            "min_confidence": 0.05,
            "depth_noise_ref_m": 4.0,
            "occlusion_penalty": 0.3,
            "bone_length_constraint": false,
            "filter": { "min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0 }
        },
        "mapping": {
            "trackers": ["waist", "left_foot", "right_foot"],
            "emit_head": true,
            "velocity_smooth": 0.5
        },
        "endpoints": [
            { "type": "osc", "params": { "host": "127.0.0.1", "port": 9123 } }
        ]
    })json");
}

} // namespace

TEST_CASE("e2e: mock nodes -> pipeline -> OSC over UDP loopback") {
    REQUIRE(mn::UdpSocket::globalInit());

    // Bind the receiver before the pipeline starts so no packets are refused.
    mn::UdpSocket rx;
    REQUIRE_MESSAGE(rx.openReceive(kPort, "127.0.0.1"), rx.lastError());

    const mn::AppConfig cfg = mn::AppConfig::fromJson(e2eConfigJson());

    mn::NodeRegistry nodeReg;
    mn::mock::registerNodes(nodeReg);
    mn::EndpointRegistry epReg;
    mn::osc::registerEndpoints(epReg);
    mn::CalibrationStore calib; // empty: extrinsics resolve from node params

    std::string error;
    auto pipeline = mn::Pipeline::build(cfg, nodeReg, epReg, calib, error);
    REQUIRE_MESSAGE(pipeline != nullptr, error);
    REQUIRE(pipeline->start());

    // Mapping order is waist, left_foot, right_foot (+ head appended), and the
    // endpoint assigns numeric slots in order of appearance, so waist is "1".
    const std::string waistPosAddr = "/tracking/trackers/1/position";
    const std::string headPosAddr = "/tracking/trackers/head/position";

    size_t packetsParsed = 0;
    std::set<std::string> addresses;
    bool allFinite = true;
    size_t waistSamples = 0;
    float waistYMin = 1e9f;
    float waistYMax = -1e9f;

    uint8_t buf[1024];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (std::chrono::steady_clock::now() < deadline) {
        const int n = rx.receive(buf, sizeof(buf), 200);
        if (n < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (n == 0)
            continue; // timeout, keep draining until the deadline
        const auto msgs = mn::osc::parsePacket(buf, static_cast<size_t>(n));
        if (msgs.empty())
            continue;
        ++packetsParsed;
        for (const auto& m : msgs) {
            addresses.insert(m.address);
            for (float f : m.floats) {
                if (!std::isfinite(f))
                    allFinite = false;
            }
            if (m.address == waistPosAddr && m.floats.size() == 3) {
                ++waistSamples;
                waistYMin = std::min(waistYMin, m.floats[1]);
                waistYMax = std::max(waistYMax, m.floats[1]);
            }
        }
    }

    pipeline->stop();
    CHECK_FALSE(pipeline->isRunning());

    // At 60 Hz ticks with 30 Hz mock nodes for ~2 s we expect hundreds of
    // packets; >= 30 leaves huge margin for slow CI.
    CHECK_MESSAGE(packetsParsed >= 30, "only parsed " << packetsParsed << " packets");
    CHECK_MESSAGE(addresses.count(waistPosAddr) == 1, "missing " << waistPosAddr);
    CHECK_MESSAGE(addresses.count(headPosAddr) == 1, "missing " << headPosAddr);
    CHECK(allFinite);

    // Unity y equals Marionette world y (only z is negated); a standing
    // ~1.7 m body keeps its waist comfortably inside [0.5, 1.4] m.
    REQUIRE(waistSamples > 0);
    CHECK_MESSAGE(waistYMin >= 0.5f, "waist y min " << waistYMin);
    CHECK_MESSAGE(waistYMax <= 1.4f, "waist y max " << waistYMax);

    const mn::Pipeline::Stats stats = pipeline->stats();
    CHECK(stats.ticks > 0);
    CHECK(stats.framesIn > 0);
    CHECK(stats.trackersOut > 0);
}
