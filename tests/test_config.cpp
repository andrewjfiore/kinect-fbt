#include <doctest/doctest.h>

#include "mn/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

using nlohmann::json;

namespace {

// The full example from docs/DESIGN.md "Config schema" (comments stripped,
// placeholder IP filled in, node type picked from the documented set).
const char* kFullConfig = R"json({
  "tick_hz": 90,
  "calibration_file": "calibration.json",
  "nodes": [ { "id": "front", "type": "mock",
               "params": { "rate_hz": 30,
                           "extrinsic": { "pos": [0.0, 0.9, -2.0], "rot": [1, 0, 0, 0] } } },
             { "id": "back", "type": "kinect_v2", "params": {} } ],
  "fusion": { "stale_seconds": 0.15, "inferred_weight": 0.25,
              "min_confidence": 0.05, "depth_noise_ref_m": 4.0,
              "occlusion_penalty": 0.3, "bone_length_constraint": true,
              "filter": { "min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0 } },
  "mapping": { "trackers": ["waist","left_foot","right_foot","chest",
                            "left_knee","right_knee","left_elbow","right_elbow"],
               "emit_head": true, "velocity_smooth": 0.5 },
  "endpoints": [ { "type": "osc",    "params": { "host": "192.168.1.50", "port": 9000 } },
                 { "type": "openvr", "params": { "host": "127.0.0.1", "port": 24190 } } ]
})json";

std::filesystem::path tempFile(const char* stem) {
    static std::mt19937_64 rng{std::random_device{}()};
    return std::filesystem::temp_directory_path() /
           (std::string(stem) + "-" + std::to_string(rng()) + ".json");
}

mn::Pose makePose(float x, float y, float z, float angleRad) {
    mn::Pose p;
    p.pos = mn::Vec3(x, y, z);
    p.rot = mn::Quat(Eigen::AngleAxisf(angleRad, mn::Vec3(0.3f, 1.0f, 0.2f).normalized()));
    return p;
}

void checkPoseApprox(const mn::Pose& a, const mn::Pose& b) {
    CHECK(a.pos.x() == doctest::Approx(b.pos.x()).epsilon(1e-4));
    CHECK(a.pos.y() == doctest::Approx(b.pos.y()).epsilon(1e-4));
    CHECK(a.pos.z() == doctest::Approx(b.pos.z()).epsilon(1e-4));
    // q and -q are the same rotation; compare |dot|.
    CHECK(std::abs(a.rot.dot(b.rot)) == doctest::Approx(1.0f).epsilon(1e-5));
}

} // namespace

TEST_CASE("AppConfig parses the full DESIGN.md example") {
    const mn::AppConfig cfg = mn::AppConfig::fromJson(json::parse(kFullConfig));

    CHECK(cfg.tickHz == doctest::Approx(90.0));
    CHECK(cfg.calibrationFile == "calibration.json");

    REQUIRE(cfg.nodes.size() == 2u);
    CHECK(cfg.nodes[0].id == "front");
    CHECK(cfg.nodes[0].type == "mock");
    CHECK(cfg.nodes[0].params.at("rate_hz").get<double>() == doctest::Approx(30.0));
    CHECK(cfg.nodes[0].params.contains("extrinsic"));
    CHECK(cfg.nodes[1].id == "back");
    CHECK(cfg.nodes[1].type == "kinect_v2");
    CHECK(cfg.nodes[1].params.is_object());

    CHECK(cfg.fusion.staleSeconds == doctest::Approx(0.15));
    CHECK(cfg.fusion.inferredWeight == doctest::Approx(0.25f));
    CHECK(cfg.fusion.minConfidence == doctest::Approx(0.05f));
    CHECK(cfg.fusion.depthNoiseRefMeters == doctest::Approx(4.0f));
    CHECK(cfg.fusion.occlusionPenalty == doctest::Approx(0.3f));
    CHECK(cfg.fusion.boneLengthConstraint == true);
    CHECK(cfg.fusion.filter.minCutoff == doctest::Approx(1.0f));
    CHECK(cfg.fusion.filter.beta == doctest::Approx(0.05f));
    CHECK(cfg.fusion.filter.dCutoff == doctest::Approx(1.0f));

    REQUIRE(cfg.mapping.roles.size() == 8u);
    CHECK(cfg.mapping.roles[0] == mn::TrackerRole::Waist);
    CHECK(cfg.mapping.roles[1] == mn::TrackerRole::LeftFoot);
    CHECK(cfg.mapping.roles[2] == mn::TrackerRole::RightFoot);
    CHECK(cfg.mapping.roles[3] == mn::TrackerRole::Chest);
    CHECK(cfg.mapping.roles[4] == mn::TrackerRole::LeftKnee);
    CHECK(cfg.mapping.roles[5] == mn::TrackerRole::RightKnee);
    CHECK(cfg.mapping.roles[6] == mn::TrackerRole::LeftElbow);
    CHECK(cfg.mapping.roles[7] == mn::TrackerRole::RightElbow);
    CHECK(cfg.mapping.emitHead == true);
    CHECK(cfg.mapping.velocitySmooth == doctest::Approx(0.5f));

    REQUIRE(cfg.endpoints.size() == 2u);
    CHECK(cfg.endpoints[0].type == "osc");
    CHECK(cfg.endpoints[0].params.at("host").get<std::string>() == "192.168.1.50");
    CHECK(cfg.endpoints[0].params.at("port").get<int>() == 9000);
    CHECK(cfg.endpoints[1].type == "openvr");
    CHECK(cfg.endpoints[1].params.at("port").get<int>() == 24190);
}

TEST_CASE("AppConfig applies defaults when keys are missing") {
    const mn::AppConfig cfg = mn::AppConfig::fromJson(json::object());
    CHECK(cfg.tickHz == doctest::Approx(90.0));
    CHECK(cfg.calibrationFile == "calibration.json");
    CHECK(cfg.nodes.empty());
    CHECK(cfg.endpoints.empty());

    CHECK(cfg.fusion.staleSeconds == doctest::Approx(0.15));
    CHECK(cfg.fusion.inferredWeight == doctest::Approx(0.25f));
    CHECK(cfg.fusion.minConfidence == doctest::Approx(0.05f));
    CHECK(cfg.fusion.depthNoiseRefMeters == doctest::Approx(4.0f));
    CHECK(cfg.fusion.occlusionPenalty == doctest::Approx(0.3f));
    CHECK(cfg.fusion.boneLengthConstraint == true);
    CHECK(cfg.fusion.filter.minCutoff == doctest::Approx(1.0f));
    CHECK(cfg.fusion.filter.beta == doctest::Approx(0.05f));
    CHECK(cfg.fusion.filter.dCutoff == doctest::Approx(1.0f));

    REQUIRE(cfg.mapping.roles.size() == 3u);
    CHECK(cfg.mapping.roles[0] == mn::TrackerRole::Waist);
    CHECK(cfg.mapping.roles[1] == mn::TrackerRole::LeftFoot);
    CHECK(cfg.mapping.roles[2] == mn::TrackerRole::RightFoot);
    CHECK(cfg.mapping.emitHead == true);
    CHECK(cfg.mapping.velocitySmooth == doctest::Approx(0.5f));

    SUBCASE("partial fusion object keeps sibling defaults") {
        const json partial = json::parse(R"({"fusion":{"stale_seconds":0.5}})");
        const mn::AppConfig c = mn::AppConfig::fromJson(partial);
        CHECK(c.fusion.staleSeconds == doctest::Approx(0.5));
        CHECK(c.fusion.inferredWeight == doctest::Approx(0.25f));
        CHECK(c.fusion.filter.beta == doctest::Approx(0.05f));
    }
}

TEST_CASE("AppConfig rejects bad input with std::runtime_error") {
    SUBCASE("unknown tracker role name") {
        const json j = json::parse(R"({"mapping":{"trackers":["waist","waste"]}})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(j), std::runtime_error);
        try {
            (void)mn::AppConfig::fromJson(j);
            FAIL("expected std::runtime_error");
        } catch (const std::runtime_error& e) {
            // Error message carries the field context.
            CHECK(std::string(e.what()).find("config.mapping.trackers[1]") != std::string::npos);
        }
    }
    SUBCASE("node missing id") {
        const json j = json::parse(R"({"nodes":[{"type":"mock"}]})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(j), std::runtime_error);
    }
    SUBCASE("node missing type") {
        const json j = json::parse(R"({"nodes":[{"id":"front"}]})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(j), std::runtime_error);
    }
    SUBCASE("endpoint missing type") {
        const json j = json::parse(R"({"endpoints":[{"params":{}}]})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(j), std::runtime_error);
    }
    SUBCASE("tick_hz must be a positive number") {
        const json zero = json::parse(R"({"tick_hz":0})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(zero), std::runtime_error);
        const json text = json::parse(R"({"tick_hz":"fast"})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(text), std::runtime_error);
    }
    SUBCASE("nodes must be an array") {
        const json j = json::parse(R"({"nodes":{}})");
        CHECK_THROWS_AS((void)mn::AppConfig::fromJson(j), std::runtime_error);
    }
}

TEST_CASE("AppConfig toJson round-trips through fromJson") {
    const mn::AppConfig cfg = mn::AppConfig::fromJson(json::parse(kFullConfig));
    const mn::AppConfig cfg2 = mn::AppConfig::fromJson(cfg.toJson());
    CHECK(cfg2.tickHz == doctest::Approx(cfg.tickHz));
    CHECK(cfg2.calibrationFile == cfg.calibrationFile);
    CHECK(cfg2.nodes.size() == cfg.nodes.size());
    CHECK(cfg2.endpoints.size() == cfg.endpoints.size());
    CHECK(cfg2.mapping.roles == cfg.mapping.roles);
    CHECK(cfg2.mapping.emitHead == cfg.mapping.emitHead);
    CHECK(cfg2.fusion.staleSeconds == doctest::Approx(cfg.fusion.staleSeconds));
    CHECK(cfg2.fusion.boneLengthConstraint == cfg.fusion.boneLengthConstraint);
}

TEST_CASE("Pose JSON round-trip") {
    const mn::Pose p = makePose(0.25f, 1.5f, -2.0f, 0.8f);
    const json j = p;
    checkPoseApprox(j.get<mn::Pose>(), p);

    SUBCASE("rot omitted defaults to identity") {
        const json noRot = json::parse(R"({"pos":[1.0,2.0,3.0]})");
        const mn::Pose r = noRot.get<mn::Pose>();
        CHECK(r.pos.x() == doctest::Approx(1.0f));
        CHECK(r.pos.y() == doctest::Approx(2.0f));
        CHECK(r.pos.z() == doctest::Approx(3.0f));
        CHECK(r.rot.w() == doctest::Approx(1.0f));
        CHECK(r.rot.x() == doctest::Approx(0.0f));
        CHECK(r.rot.y() == doctest::Approx(0.0f));
        CHECK(r.rot.z() == doctest::Approx(0.0f));
    }
    SUBCASE("non-unit rot is normalized on load") {
        const json scaled = json::parse(R"({"pos":[0,0,0],"rot":[2.0,0.0,0.0,0.0]})");
        const mn::Pose r = scaled.get<mn::Pose>();
        CHECK(r.rot.norm() == doctest::Approx(1.0f));
        CHECK(r.rot.w() == doctest::Approx(1.0f));
    }
    SUBCASE("malformed poses throw") {
        const json noPos = json::parse(R"({"rot":[1,0,0,0]})");
        CHECK_THROWS_AS((void)noPos.get<mn::Pose>(), std::runtime_error);
        const json shortPos = json::parse(R"({"pos":[1,2]})");
        CHECK_THROWS_AS((void)shortPos.get<mn::Pose>(), std::runtime_error);
        const json zeroQuat = json::parse(R"({"pos":[0,0,0],"rot":[0,0,0,0]})");
        CHECK_THROWS_AS((void)zeroQuat.get<mn::Pose>(), std::runtime_error);
    }
}

TEST_CASE("CalibrationStore round-trips through a temp file") {
    const std::filesystem::path path = tempFile("mn-calib-roundtrip");
    const mn::Pose front = makePose(0.0f, 0.9f, -2.0f, 0.3f);
    const mn::Pose back = makePose(0.1f, 1.0f, 2.0f, 2.9f);
    const mn::Pose anchor = makePose(-0.4f, 0.0f, 0.7f, 1.2f);

    mn::CalibrationStore a;
    a.setNodeExtrinsic("front", front);
    a.setNodeExtrinsic("back", back);
    mn::BodyModel bm;
    bm.valid = true;
    bm.boneLengthToParent[static_cast<size_t>(mn::Joint::KneeL)] = 0.45f;
    bm.boneLengthToParent[static_cast<size_t>(mn::Joint::AnkleL)] = 0.41f;
    bm.boneLengthToParent[static_cast<size_t>(mn::Joint::Neck)] = 0.09f;
    a.setBodyModel(bm);
    a.setWorldAnchor(anchor);
    REQUIRE(a.save(path.string()));

    mn::CalibrationStore b;
    REQUIRE(b.load(path.string()));
    REQUIRE(b.nodeExtrinsic("front").has_value());
    checkPoseApprox(*b.nodeExtrinsic("front"), front);
    REQUIRE(b.nodeExtrinsic("back").has_value());
    checkPoseApprox(*b.nodeExtrinsic("back"), back);
    CHECK_FALSE(b.nodeExtrinsic("missing").has_value());

    CHECK(b.bodyModel().valid);
    CHECK(b.bodyModel().boneLengthToParent[static_cast<size_t>(mn::Joint::KneeL)] ==
          doctest::Approx(0.45f));
    CHECK(b.bodyModel().boneLengthToParent[static_cast<size_t>(mn::Joint::AnkleL)] ==
          doctest::Approx(0.41f));
    CHECK(b.bodyModel().boneLengthToParent[static_cast<size_t>(mn::Joint::Neck)] ==
          doctest::Approx(0.09f));
    CHECK(b.bodyModel().boneLengthToParent[static_cast<size_t>(mn::Joint::WristR)] ==
          doctest::Approx(0.0f));

    REQUIRE(b.worldAnchor().has_value());
    checkPoseApprox(*b.worldAnchor(), anchor);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("CalibrationStore missing file loads as an empty ok store") {
    const std::filesystem::path path = tempFile("mn-calib-missing");
    mn::CalibrationStore s;
    CHECK(s.load(path.string()));
    CHECK_FALSE(s.nodeExtrinsic("front").has_value());
    CHECK_FALSE(s.bodyModel().valid);
    CHECK_FALSE(s.worldAnchor().has_value());
}

TEST_CASE("CalibrationStore malformed file fails to load") {
    const std::filesystem::path path = tempFile("mn-calib-bad");
    {
        std::ofstream out(path);
        REQUIRE(static_cast<bool>(out));
        out << "{ this is not json";
    }
    mn::CalibrationStore s;
    CHECK_FALSE(s.load(path.string()));
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
