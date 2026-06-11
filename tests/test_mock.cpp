#include <doctest/doctest.h>

#include "mn_mock/mock.hpp"

#include "mn/clock.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace mn;

namespace {

std::array<float, kJointCount> boneLengths(const SkeletonFrame& f) {
    std::array<float, kJointCount> out{};
    for (size_t i = 0; i < kJointCount; ++i) {
        const auto j = static_cast<Joint>(i);
        if (j == Joint::Hips)
            continue; // root
        out[i] = (f[j].pos - f[jointParent(j)].pos).norm();
    }
    return out;
}

std::string tempJsonlPath(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

TEST_CASE("mock: groundTruth is deterministic") {
    for (const char* pattern : {"walk_in_place", "tpose", "sway"}) {
        const SkeletonFrame a = mock::groundTruth(0.37, pattern);
        const SkeletonFrame b = mock::groundTruth(0.37, pattern);
        CHECK(a.timestamp == b.timestamp);
        CHECK(a.hasBody);
        for (size_t i = 0; i < kJointCount; ++i) {
            CHECK(a.joints[i].pos.x() == b.joints[i].pos.x());
            CHECK(a.joints[i].pos.y() == b.joints[i].pos.y());
            CHECK(a.joints[i].pos.z() == b.joints[i].pos.z());
            CHECK(a.joints[i].confidence == 1.0f);
            CHECK(a.joints[i].state == TrackState::Tracked);
            CHECK_FALSE(a.joints[i].hasRot);
        }
    }
}

TEST_CASE("mock: groundTruth bone lengths constant across t for walk_in_place") {
    const auto ref = boneLengths(mock::groundTruth(0.0, "walk_in_place"));
    for (size_t i = 0; i < kJointCount; ++i) {
        if (static_cast<Joint>(i) == Joint::Hips)
            continue;
        CHECK(ref[i] > 0.02f); // sane, non-degenerate skeleton
    }
    for (const double t : {0.3, 1.7}) {
        const auto cur = boneLengths(mock::groundTruth(t, "walk_in_place"));
        for (size_t i = 0; i < kJointCount; ++i) {
            if (static_cast<Joint>(i) == Joint::Hips)
                continue;
            CHECK(cur[i] == doctest::Approx(ref[i]).epsilon(1e-4));
        }
    }
}

TEST_CASE("mock: groundTruth left/right matches the facing -Z convention") {
    for (const double t : {0.0, 0.3, 1.7}) {
        const SkeletonFrame f = mock::groundTruth(t, "walk_in_place");
        CHECK(f[Joint::AnkleL].pos.x() < f[Joint::AnkleR].pos.x());
        CHECK(f[Joint::ShoulderL].pos.x() < f[Joint::ShoulderR].pos.x());
        CHECK(f[Joint::HipL].pos.x() < f[Joint::HipR].pos.x());
    }
    const SkeletonFrame f0 = mock::groundTruth(0.0, "tpose");
    CHECK(f0[Joint::Hips].pos.y() == doctest::Approx(0.95).epsilon(0.05));
    CHECK(f0[Joint::Head].pos.y() > 1.5f); // ~1.70 m body
    CHECK(f0[Joint::Head].pos.y() < 1.8f);
}

TEST_CASE("mock: JsonlRecorder -> loadJsonl round-trip") {
    const std::string path = tempJsonlPath("mn_test_mock_roundtrip.jsonl");
    std::vector<SkeletonFrame> written;
    for (const double t : {0.0, 0.3, 1.7})
        written.push_back(mock::groundTruth(t, "walk_in_place"));
    // Exercise non-default per-joint fields too.
    written[1].joints[0].state = TrackState::Inferred;
    written[1].joints[0].confidence = 0.42f;
    written[2].hasBody = false;

    {
        mock::JsonlRecorder rec;
        REQUIRE(rec.open(path));
        for (const auto& f : written)
            rec.write(f);
        CHECK(rec.frameCount() == written.size());
        rec.close();
    }

    std::vector<SkeletonFrame> loaded;
    REQUIRE(mock::loadJsonl(path, loaded));
    REQUIRE(loaded.size() == written.size());
    for (size_t k = 0; k < written.size(); ++k) {
        CHECK(loaded[k].timestamp == doctest::Approx(written[k].timestamp).epsilon(1e-9));
        CHECK(loaded[k].hasBody == written[k].hasBody);
        for (size_t i = 0; i < kJointCount; ++i) {
            const JointSample& w = written[k].joints[i];
            const JointSample& l = loaded[k].joints[i];
            CHECK(l.pos.x() == doctest::Approx(w.pos.x()).epsilon(1e-6));
            CHECK(l.pos.y() == doctest::Approx(w.pos.y()).epsilon(1e-6));
            CHECK(l.pos.z() == doctest::Approx(w.pos.z()).epsilon(1e-6));
            CHECK(l.confidence == doctest::Approx(w.confidence).epsilon(1e-6));
            CHECK(l.state == w.state);
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::vector<SkeletonFrame> none;
    CHECK_FALSE(mock::loadJsonl(path, none)); // file is gone now
}

TEST_CASE("mock: factory validates params") {
    NodeRegistry reg;
    mock::registerNodes(reg);
    std::string error;

    auto bad = reg.create("mock", "m", nlohmann::json{{"pattern", "cartwheel"}}, error);
    CHECK(bad == nullptr);
    CHECK_FALSE(error.empty());

    error.clear();
    bad = reg.create("mock", "m", nlohmann::json{{"rate_hz", -5.0}}, error);
    CHECK(bad == nullptr);
    CHECK_FALSE(error.empty());

    error.clear();
    bad = reg.create("replay", "r", nlohmann::json{{"file", "definitely/missing.jsonl"}}, error);
    CHECK(bad == nullptr);
    CHECK_FALSE(error.empty());
}

TEST_CASE("mock: mock node emits frames at rate_hz and stops promptly") {
    NodeRegistry reg;
    mock::registerNodes(reg);
    std::string error;
    auto node = reg.create("mock", "m1",
                           nlohmann::json{{"pattern", "tpose"}, {"rate_hz", 30.0}}, error);
    REQUIRE(node != nullptr);

    std::atomic<int> count{0};
    REQUIRE(node->start([&count](const NodeDescriptor& d, const SkeletonFrame& f) {
        (void)d;
        if (f.hasBody)
            ++count;
    }));
    CHECK(node->isRunning());
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    const auto t0 = std::chrono::steady_clock::now();
    node->stop();
    const double stopMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(count.load() >= 5);
    CHECK_FALSE(node->isRunning());
    CHECK(stopMs < 250.0);
}

TEST_CASE("mock: replay node with loop=false ends by itself") {
    const std::string path = tempJsonlPath("mn_test_mock_replay.jsonl");
    {
        mock::JsonlRecorder rec;
        REQUIRE(rec.open(path));
        for (int i = 0; i < 5; ++i)
            rec.write(mock::groundTruth(0.02 * i, "tpose"));
        rec.close();
    }

    NodeRegistry reg;
    mock::registerNodes(reg);
    std::string error;
    auto node = reg.create(
        "replay", "r1", nlohmann::json{{"file", path}, {"loop", false}, {"speed", 1.0}}, error);
    REQUIRE(node != nullptr);

    const double startSec = nowSeconds();
    std::atomic<int> count{0};
    std::atomic<double> minTs{1e300};
    REQUIRE(node->start([&](const NodeDescriptor& d, const SkeletonFrame& f) {
        (void)d;
        ++count;
        if (f.timestamp < minTs.load())
            minTs.store(f.timestamp); // single capture thread writes this
    }));

    bool ended = false;
    for (int i = 0; i < 200; ++i) {
        if (!node->isRunning()) {
            ended = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(ended);
    CHECK(count.load() == 5);
    CHECK(minTs.load() >= startSec - 0.05); // restamped to emission time, not stored t
    node->stop(); // joins the finished thread; must be safe after natural end

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
