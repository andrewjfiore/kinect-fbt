// Dashboard HTTP server test: two mock capture nodes -> pipeline ->
// DashboardServer on loopback, exercised end-to-end with an httplib client:
// status / skeleton / events / index, then a full pair-calibration job.

// mn headers (Eigen) BEFORE httplib: glibc resolver headers pulled in by
// httplib define macros that corrupt Eigen's templates when Eigen is parsed
// afterwards. Safe on Windows too: no mn HEADER includes windows.h.
#include "mn/config.hpp"
#include "mn/pipeline.hpp"
#include "mn_dashboard/dashboard.hpp"
#include "mn_mock/mock.hpp"

#include <httplib.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>

namespace {

constexpr uint16_t kPort = 18211;

nlohmann::json dashboardConfigJson() {
    return nlohmann::json::parse(R"json({
        "tick_hz": 60,
        "calibration_file": "test_dashboard_calibration.json",
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
        "endpoints": []
    })json");
}

nlohmann::json getJson(httplib::Client& cli, const std::string& path) {
    auto res = cli.Get(path);
    REQUIRE_MESSAGE(res, "GET " << path << " failed: " << httplib::to_string(res.error()));
    REQUIRE_MESSAGE(res->status == 200, "GET " << path << " -> status " << res->status);
    return nlohmann::json::parse(res->body);
}

} // namespace

TEST_CASE("dashboard: HTTP API over loopback with a live pipeline") {
    const mn::AppConfig cfg = mn::AppConfig::fromJson(dashboardConfigJson());
    const std::string calibPath = "test_dashboard_calibration.json";
    std::remove(calibPath.c_str());

    mn::NodeRegistry nodeReg;
    mn::mock::registerNodes(nodeReg);
    mn::EndpointRegistry epReg; // no endpoints
    mn::CalibrationStore store; // empty: extrinsics resolve from node params

    std::string error;
    auto pipeline = mn::Pipeline::build(cfg, nodeReg, epReg, store, error);
    REQUIRE_MESSAGE(pipeline != nullptr, error);
    REQUIRE(pipeline->start());

    mn::dash::Options opt;
    opt.bind = "127.0.0.1";
    opt.port = kPort;
    opt.capabilities = {{"kinect_v2", false}, {"kinect_v1", false}, {"openvr_client", false}};
    mn::dash::DashboardServer server(*pipeline, store, calibPath, opt);
    REQUIRE_MESSAGE(server.start(), "dashboard failed to bind 127.0.0.1:"
                                        << kPort << ": " << server.lastError());
    CHECK(server.isRunning());
    CHECK(server.url() == "http://127.0.0.1:18211");

    httplib::Client cli("127.0.0.1", kPort);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(5, 0);

    // --- /api/skeleton: wait for the first fused body (~0.5 s), then verify.
    nlohmann::json skel;
    {
        bool fused = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (std::chrono::steady_clock::now() < deadline) {
            skel = getJson(cli, "/api/skeleton");
            if (skel["has_body"].get<bool>()) {
                fused = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        REQUIRE_MESSAGE(fused, "no fused body within 3 s");
    }
    CHECK(skel["joints"].size() == 19);
    for (const auto& [name, js] : skel["joints"].items()) {
        REQUIRE_MESSAGE(js.contains("p"), "joint " << name << " is missing p");
        const auto& p = js.at("p");
        REQUIRE_MESSAGE(p.is_array(), "joint " << name << " p is not an array");
        REQUIRE(p.size() == 3);
        for (const auto& v : p)
            CHECK_MESSAGE(std::isfinite(v.get<double>()), "joint " << name << " non-finite pos");
        CHECK(std::isfinite(js.at("c").get<double>()));
        CHECK(js.at("s").is_number_integer());
    }
    CHECK(skel["bones"].size() == 18); // every joint but the Hips root
    CHECK(skel["sensors"].size() == 2);

    // --- /api/status
    {
        auto res = cli.Get("/api/status");
        REQUIRE(res);
        CHECK(res->get_header_value("Cache-Control") == "no-store");
        const nlohmann::json st = nlohmann::json::parse(res->body);
        CHECK(st["pipeline"]["running"].get<bool>());
        CHECK(st["nodes"].size() == 2);
        CHECK_MESSAGE(st["pipeline"]["fused_pct"].get<double>() > 0.0,
                      "fused_pct " << st["pipeline"]["fused_pct"].get<double>());
        CHECK(st["app"]["name"] == "marionette");
        CHECK(st["calibration"]["extrinsics"].size() == 0); // store starts empty
    }

    // --- GET / serves the embedded UI
    {
        auto res = cli.Get("/");
        REQUIRE(res);
        CHECK(res->status == 200);
        CHECK(res->get_header_value("Content-Type").find("text/html") != std::string::npos);
        CHECK_MESSAGE(res->body.find("<html") != std::string::npos,
                      "index body does not contain <html");
    }

    // --- /api/events
    {
        const nlohmann::json ev = getJson(cli, "/api/events");
        REQUIRE(ev.contains("latest_seq"));
        CHECK(ev["latest_seq"].get<long long>() >= 0);
        CHECK(ev["events"].is_array());
        CHECK(ev["counts"].contains("error"));
    }

    // --- pair calibration job: front (reference) vs back (target)
    {
        const nlohmann::json body{{"reference", "front"},
                                  {"target", "back"},
                                  {"min_samples", 60},
                                  {"max_seconds", 10}};
        auto res = cli.Post("/api/calibrate/pair", body.dump(), "application/json");
        REQUIRE_MESSAGE(res, "POST /api/calibrate/pair failed: "
                                 << httplib::to_string(res.error()));
        const nlohmann::json started = nlohmann::json::parse(res->body);
        REQUIRE_MESSAGE(started["ok"].get<bool>(),
                        "pair job rejected: " << started["error"].get<std::string>());
    }
    nlohmann::json jobStatus;
    {
        bool finished = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (std::chrono::steady_clock::now() < deadline) {
            jobStatus = getJson(cli, "/api/calibrate/status");
            if (jobStatus["done"].get<bool>()) {
                finished = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        REQUIRE_MESSAGE(finished, "pair calibration did not finish within 15 s");
    }
    CHECK(jobStatus["kind"] == "pair");
    CHECK_FALSE(jobStatus["active"].get<bool>());
    CHECK_MESSAGE(jobStatus["ok"].get<bool>(),
                  "pair job failed: " << jobStatus["message"].get<std::string>());
    CHECK_MESSAGE(jobStatus["rmse_cm"].get<double>() < 5.0,
                  "rmse_cm " << jobStatus["rmse_cm"].get<double>());

    // --- the solved extrinsic for "back" is now in the calibration store
    {
        const nlohmann::json st = getJson(cli, "/api/status");
        bool hasBack = false;
        for (const auto& id : st["calibration"]["extrinsics"])
            hasBack = hasBack || (id.get<std::string>() == "back");
        CHECK_MESSAGE(hasBack, "calibration.extrinsics does not list 'back'");
    }

    server.stop();
    CHECK_FALSE(server.isRunning());
    pipeline->stop();
    std::remove(calibPath.c_str());
}
