#include "mn/calibration.hpp"

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

mn::Quat randomRotation(std::mt19937& rng) {
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const mn::Vec3 axis = mn::safeNormalized(mn::Vec3(gauss(rng), gauss(rng), gauss(rng)),
                                             mn::Vec3::UnitX());
    std::uniform_real_distribution<float> angle(-3.1f, 3.1f);
    return mn::Quat(Eigen::AngleAxisf(angle(rng), axis)).normalized();
}

mn::Vec3 randomVec(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    return mn::Vec3(u(rng), u(rng), u(rng));
}

// Joints used by the default PairCalibrationSession options, with body-relative
// offsets that span all three axes (head up, wrists out, ankles down).
const std::array<std::pair<mn::Joint, mn::Vec3>, 6> kBodyShape = {{
    {mn::Joint::Head, mn::Vec3(0.0f, 0.65f, 0.0f)},
    {mn::Joint::Hips, mn::Vec3(0.0f, 0.0f, 0.0f)},
    {mn::Joint::WristL, mn::Vec3(-0.35f, 0.25f, 0.10f)},
    {mn::Joint::WristR, mn::Vec3(0.35f, 0.25f, 0.05f)},
    {mn::Joint::AnkleL, mn::Vec3(-0.12f, -0.85f, 0.02f)},
    {mn::Joint::AnkleR, mn::Vec3(0.12f, -0.85f, -0.03f)},
}};

mn::SkeletonFrame frameWithJoints(const std::array<mn::Vec3, 6>& positions, double timestamp,
                                  float confidence = 0.9f) {
    mn::SkeletonFrame frame;
    frame.timestamp = timestamp;
    frame.hasBody = true;
    for (size_t i = 0; i < kBodyShape.size(); ++i) {
        mn::JointSample& js = frame[kBodyShape[i].first];
        js.pos = positions[i];
        js.confidence = confidence;
        js.state = mn::TrackState::Tracked;
    }
    return frame;
}

} // namespace

TEST_CASE("solveRigid recovers random SE(3) transforms exactly") {
    std::mt19937 rng(7u);
    for (int k = 0; k < 10; ++k) {
        const mn::Quat q = randomRotation(rng);
        const mn::Vec3 t = randomVec(rng, -2.0f, 2.0f);
        const mn::Pose truth{t, q};

        std::vector<mn::Vec3> src;
        std::vector<mn::Vec3> dst;
        for (int i = 0; i < 50; ++i) {
            const mn::Vec3 p = randomVec(rng, -1.0f, 1.0f);
            src.push_back(p);
            dst.push_back(truth.apply(p));
        }

        const mn::RigidFit fit = mn::solveRigid(src, dst);
        REQUIRE(fit.ok);
        CHECK(fit.samples == 50u);
        CHECK(fit.rmse < 1e-4f);
        CHECK((fit.transform.pos - t).norm() < 1e-3f);
        // quatAngle bottoms out around ~5e-4 rad in float precision.
        CHECK(mn::quatAngle(fit.transform.rot, q) < 5e-3f);
    }
}

TEST_CASE("solveRigid with 5mm gaussian noise stays within tolerance") {
    std::mt19937 rng(99u);
    std::normal_distribution<float> noise(0.0f, 0.005f);
    for (int k = 0; k < 3; ++k) {
        const mn::Quat q = randomRotation(rng);
        const mn::Vec3 t = randomVec(rng, -2.0f, 2.0f);
        const mn::Pose truth{t, q};

        std::vector<mn::Vec3> src;
        std::vector<mn::Vec3> dst;
        for (int i = 0; i < 50; ++i) {
            const mn::Vec3 p = randomVec(rng, -1.0f, 1.0f);
            src.push_back(p);
            dst.push_back(truth.apply(p) + mn::Vec3(noise(rng), noise(rng), noise(rng)));
        }

        const mn::RigidFit fit = mn::solveRigid(src, dst);
        REQUIRE(fit.ok);
        CHECK((fit.transform.pos - t).norm() < 0.02f);                  // < 2 cm
        CHECK(mn::quatAngle(fit.transform.rot, q) < 2.0f * kDegToRad);  // < 2 deg
        CHECK(fit.rmse < 0.02f);
    }
}

TEST_CASE("solveRigid rejects degenerate input") {
    SUBCASE("fewer than 3 pairs") {
        std::vector<mn::Vec3> src;
        std::vector<mn::Vec3> dst;
        src.push_back(mn::Vec3(0.0f, 0.0f, 0.0f));
        src.push_back(mn::Vec3(1.0f, 0.0f, 0.0f));
        dst = src;
        const mn::RigidFit fit = mn::solveRigid(src, dst);
        CHECK_FALSE(fit.ok);
        CHECK(fit.samples == 2u);
    }

    SUBCASE("mismatched sizes") {
        std::vector<mn::Vec3> src(5, mn::Vec3::Zero());
        std::vector<mn::Vec3> dst(4, mn::Vec3::Zero());
        CHECK_FALSE(mn::solveRigid(src, dst).ok);
    }

    SUBCASE("collinear points") {
        const mn::Pose truth{mn::Vec3(0.3f, -0.2f, 1.0f),
                             mn::Quat(Eigen::AngleAxisf(0.8f, mn::Vec3::UnitY()))};
        const mn::Vec3 origin(0.1f, 0.2f, -0.3f);
        const mn::Vec3 dir = mn::Vec3(1.0f, 2.0f, 0.5f).normalized();
        std::vector<mn::Vec3> src;
        std::vector<mn::Vec3> dst;
        for (int i = 0; i < 20; ++i) {
            const mn::Vec3 p = origin + static_cast<float>(i) * 0.05f * dir;
            src.push_back(p);
            dst.push_back(truth.apply(p));
        }
        CHECK_FALSE(mn::solveRigid(src, dst).ok);
    }

    SUBCASE("coincident points") {
        std::vector<mn::Vec3> src(10, mn::Vec3(0.5f, 0.5f, 0.5f));
        std::vector<mn::Vec3> dst(10, mn::Vec3(1.0f, 0.0f, 0.0f));
        CHECK_FALSE(mn::solveRigid(src, dst).ok);
    }
}

TEST_CASE("PairCalibrationSession recovers a known relative pose end-to-end") {
    // Truth: target-local -> reference-local.
    const mn::Quat relRot =
        (mn::Quat(Eigen::AngleAxisf(90.0f * kDegToRad, mn::Vec3::UnitY())) *
         mn::Quat(Eigen::AngleAxisf(5.0f * kDegToRad, mn::Vec3::UnitX())))
            .normalized();
    const mn::Pose truth{mn::Vec3(1.2f, 0.1f, -0.8f), relRot};
    const mn::Pose truthInv = truth.inverse();

    // Defaults except the stillness gate: this test teleports the body each
    // frame on purpose (full-rank cloud), which a speed gate must reject.
    mn::PairCalibrationSession::Options e2eOpt;
    e2eOpt.maxJointSpeed = 0.0f;
    mn::PairCalibrationSession session(e2eOpt);

    // Before enough samples accumulate, solve() must fail.
    CHECK(session.sampleCount() == 0u);
    CHECK_FALSE(session.solve().ok);

    std::mt19937 rng(123u);
    const int frames = 40; // 40 frames x 6 joints = 240 >= 200 samples
    for (int k = 0; k < frames; ++k) {
        const mn::Vec3 root =
            mn::Vec3(0.0f, 1.0f, 1.5f) + randomVec(rng, -0.5f, 0.5f);
        std::array<mn::Vec3, 6> refPos;
        std::array<mn::Vec3, 6> tgtPos;
        for (size_t i = 0; i < kBodyShape.size(); ++i) {
            // Per-frame perturbation keeps the accumulated cloud full-rank.
            const mn::Vec3 p = root + kBodyShape[i].second + randomVec(rng, -0.05f, 0.05f);
            refPos[i] = p;                    // reference-local observation
            tgtPos[i] = truthInv.apply(p);    // target-local observation
        }
        const double t = 0.033 * static_cast<double>(k);
        session.addFramePair(frameWithJoints(refPos, t), frameWithJoints(tgtPos, t + 0.01));
    }

    CHECK(session.sampleCount() == static_cast<size_t>(frames) * kBodyShape.size());

    const mn::RigidFit fit = session.solve();
    REQUIRE(fit.ok);
    CHECK(fit.samples == session.sampleCount());
    CHECK(fit.rmse < 1e-4f);
    CHECK((fit.transform.pos - truth.pos).norm() < 1e-3f);
    CHECK(mn::quatAngle(fit.transform.rot, truth.rot) < 5e-3f);

    // Round trip: the fit maps target-local points into reference-local.
    const mn::Vec3 probe(0.2f, 1.3f, 0.7f);
    CHECK((fit.transform.apply(truthInv.apply(probe)) - probe).norm() < 1e-3f);
}

TEST_CASE("PairCalibrationSession gates samples on time, confidence, and state") {
    mn::PairCalibrationSession::Options gateOpt;
    gateOpt.maxJointSpeed = 0.0f; // single-frame subcases cannot establish stillness
    mn::PairCalibrationSession session(gateOpt);
    std::array<mn::Vec3, 6> pos;
    for (size_t i = 0; i < kBodyShape.size(); ++i) {
        pos[i] = mn::Vec3(1.0f, 1.0f, 1.0f) + kBodyShape[i].second;
    }

    SUBCASE("frames outside the pairing window add nothing") {
        session.addFramePair(frameWithJoints(pos, 0.0), frameWithJoints(pos, 0.2));
        CHECK(session.sampleCount() == 0u);
    }

    SUBCASE("frames without a body add nothing") {
        mn::SkeletonFrame empty;
        empty.timestamp = 0.0;
        empty.hasBody = false;
        session.addFramePair(frameWithJoints(pos, 0.0), empty);
        CHECK(session.sampleCount() == 0u);
    }

    SUBCASE("low-confidence joints are skipped in either view") {
        mn::SkeletonFrame tgt = frameWithJoints(pos, 0.01);
        tgt[mn::Joint::WristL].confidence = 0.2f; // below default 0.5
        session.addFramePair(frameWithJoints(pos, 0.0), tgt);
        CHECK(session.sampleCount() == 5u);
    }

    SUBCASE("non-Tracked joints are skipped even with high confidence") {
        mn::SkeletonFrame ref = frameWithJoints(pos, 0.0);
        ref[mn::Joint::Head].state = mn::TrackState::Inferred;
        session.addFramePair(ref, frameWithJoints(pos, 0.01));
        CHECK(session.sampleCount() == 5u);
    }
}

TEST_CASE("PairCalibrationSession stillness gate") {
    mn::PairCalibrationSession session; // default maxJointSpeed 0.2 m/s

    std::array<mn::Vec3, 6> pos;
    for (size_t i = 0; i < kBodyShape.size(); ++i) {
        pos[i] = mn::Vec3(0.5f, 1.0f, 2.0f) + kBodyShape[i].second;
    }

    SUBCASE("still body accumulates after the second distinct frame") {
        for (int k = 0; k < 5; ++k) {
            const double t = 0.033 * static_cast<double>(k);
            session.addFramePair(frameWithJoints(pos, t), frameWithJoints(pos, t + 0.01));
        }
        // Frame 0 establishes tracks (not still yet); frames 1..4 contribute.
        CHECK(session.sampleCount() == 4u * kBodyShape.size());
    }

    SUBCASE("fast motion is rejected") {
        for (int k = 0; k < 5; ++k) {
            const double t = 0.033 * static_cast<double>(k);
            std::array<mn::Vec3, 6> moved = pos;
            for (auto& p : moved) {
                p.x() += 0.05f * static_cast<float>(k); // ~1.5 m/s
            }
            session.addFramePair(frameWithJoints(moved, t), frameWithJoints(moved, t + 0.01));
        }
        CHECK(session.sampleCount() == 0u);
    }

    SUBCASE("slow drift passes the gate") {
        for (int k = 0; k < 5; ++k) {
            const double t = 0.033 * static_cast<double>(k);
            std::array<mn::Vec3, 6> moved = pos;
            for (auto& p : moved) {
                p.x() += 0.003f * static_cast<float>(k); // ~0.09 m/s
            }
            session.addFramePair(frameWithJoints(moved, t), frameWithJoints(moved, t + 0.01));
        }
        CHECK(session.sampleCount() == 4u * kBodyShape.size());
    }
}

TEST_CASE("PairCalibrationSession trimmed solve survives garbage pairs") {
    const mn::Pose truth{mn::Vec3(0.8f, -0.2f, 1.1f),
                         mn::Quat(Eigen::AngleAxisf(120.0f * kDegToRad, mn::Vec3::UnitY()))
                             .normalized()};
    const mn::Pose truthInv = truth.inverse();

    auto feed = [&](mn::PairCalibrationSession& session) {
        std::mt19937 rng(77u);
        for (int k = 0; k < 60; ++k) {
            const mn::Vec3 root = mn::Vec3(0.0f, 1.0f, 1.5f) + randomVec(rng, -0.5f, 0.5f);
            std::array<mn::Vec3, 6> refPos;
            std::array<mn::Vec3, 6> tgtPos;
            for (size_t i = 0; i < kBodyShape.size(); ++i) {
                const mn::Vec3 p = root + kBodyShape[i].second + randomVec(rng, -0.03f, 0.03f);
                refPos[i] = p;
                tgtPos[i] = truthInv.apply(p);
            }
            // Every 6th frame the target's WristL estimate is garbage (the
            // kind of sporadic misfire a real Kinect v1 produces).
            if (k % 6 == 0) {
                tgtPos[2] += mn::Vec3(0.9f, -0.5f, 0.4f);
            }
            const double t = 0.033 * static_cast<double>(k);
            session.addFramePair(frameWithJoints(refPos, t), frameWithJoints(tgtPos, t + 0.01));
        }
    };

    mn::PairCalibrationSession::Options opt;
    opt.maxJointSpeed = 0.0f; // teleporting synthetic cloud
    opt.trimOutliers = true;
    mn::PairCalibrationSession trimmed(opt);
    feed(trimmed);
    const mn::RigidFit good = trimmed.solve();
    REQUIRE(good.ok);
    CHECK(good.rmse < 0.04f);
    CHECK((good.transform.pos - truth.pos).norm() < 0.05f);
    CHECK(mn::quatAngle(good.transform.rot, truth.rot) < 0.05f);

    opt.trimOutliers = false;
    mn::PairCalibrationSession raw(opt);
    feed(raw);
    const mn::RigidFit bad = raw.solve();
    REQUIRE(bad.ok);
    CHECK(bad.rmse > good.rmse * 2.0f); // trimming visibly tightens the fit
}

TEST_CASE("solveAnchor maps world points into the external frame") {
    std::mt19937 rng(2026u);
    const mn::Quat q = randomRotation(rng);
    const mn::Vec3 t = randomVec(rng, -1.0f, 1.0f);
    const mn::Pose anchor{t, q}; // world -> external

    std::vector<mn::Vec3> world;
    std::vector<mn::Vec3> external;
    for (int i = 0; i < 25; ++i) {
        const mn::Vec3 p = randomVec(rng, -1.5f, 1.5f);
        world.push_back(p);
        external.push_back(anchor.apply(p));
    }

    const mn::RigidFit fit = mn::solveAnchor(world, external);
    REQUIRE(fit.ok);
    CHECK(fit.rmse < 1e-4f);
    CHECK((fit.transform.pos - t).norm() < 1e-3f);
    CHECK(mn::quatAngle(fit.transform.rot, q) < 5e-3f);
}
