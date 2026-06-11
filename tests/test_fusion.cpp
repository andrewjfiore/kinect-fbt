#include <doctest/doctest.h>

#include "mn/fusion.hpp"

#include <cmath>
#include <numbers>
#include <random>

using namespace mn;

namespace {

// Ground-truth standing body at the world origin, facing world -Z (toward the
// front sensor). Person's right is world +X.
SkeletonFrame standingBody(double timestamp) {
    SkeletonFrame f;
    f.timestamp = timestamp;
    f.hasBody = true;
    auto set = [&](Joint j, float x, float y, float z) {
        JointSample& s = f[j];
        s.pos = Vec3(x, y, z);
        s.confidence = 1.0f;
        s.state = TrackState::Tracked;
    };
    set(Joint::Hips, 0.00f, 0.95f, 0.00f);
    set(Joint::Spine, 0.00f, 1.10f, 0.00f);
    set(Joint::Chest, 0.00f, 1.25f, 0.00f);
    set(Joint::Neck, 0.00f, 1.45f, 0.00f);
    set(Joint::Head, 0.00f, 1.60f, 0.00f);
    set(Joint::ShoulderL, -0.20f, 1.40f, 0.00f);
    set(Joint::ElbowL, -0.28f, 1.12f, 0.00f);
    set(Joint::WristL, -0.30f, 0.86f, 0.00f);
    set(Joint::ShoulderR, 0.20f, 1.40f, 0.00f);
    set(Joint::ElbowR, 0.28f, 1.12f, 0.00f);
    set(Joint::WristR, 0.30f, 0.86f, 0.00f);
    set(Joint::HipL, -0.10f, 0.90f, 0.00f);
    set(Joint::KneeL, -0.11f, 0.50f, 0.00f);
    set(Joint::AnkleL, -0.12f, 0.10f, 0.00f);
    set(Joint::FootL, -0.12f, 0.03f, -0.15f);
    set(Joint::HipR, 0.10f, 0.90f, 0.00f);
    set(Joint::KneeR, 0.11f, 0.50f, 0.00f);
    set(Joint::AnkleR, 0.12f, 0.10f, 0.00f);
    set(Joint::FootR, 0.12f, 0.03f, -0.15f);
    return f;
}

// Front sensor at world (0, 1, -2.5) looking along +Z toward the user; its
// node-local axes coincide with world axes, so the extrinsic is a pure
// translation.
Pose frontExtrinsic() {
    return Pose{Vec3(0.0f, 1.0f, -2.5f), Quat::Identity()};
}

// Back sensor at world (0, 1, +2.5) looking along -Z toward the user: local
// +Z (toward user) maps to world -Z, i.e. a 180 degree yaw.
Pose backExtrinsic() {
    const Quat r(Eigen::AngleAxisf(std::numbers::pi_v<float>, Vec3::UnitY()));
    return Pose{Vec3(0.0f, 1.0f, 2.5f), r};
}

SkeletonFrame toLocal(const SkeletonFrame& world, const Pose& extrinsic) {
    SkeletonFrame f = world;
    const Pose inv = extrinsic.inverse();
    for (JointSample& s : f.joints)
        s.pos = inv.apply(s.pos);
    return f;
}

SkeletonFrame withNoise(const SkeletonFrame& f, std::mt19937& rng, float sigma) {
    std::normal_distribution<float> n(0.0f, sigma);
    SkeletonFrame out = f;
    for (JointSample& s : out.joints)
        s.pos += Vec3(n(rng), n(rng), n(rng));
    return out;
}

} // namespace

TEST_CASE("fusion: two noisy views beat one") {
    std::mt19937 rng(1234);
    const double t = 100.0;
    const SkeletonFrame truth = standingBody(t);
    const Pose front = frontExtrinsic();
    const Pose back = backExtrinsic();

    double fusedErr = 0.0;
    double singleErr = 0.0;
    int samples = 0;
    for (int trial = 0; trial < 10; ++trial) {
        FusionEngine eng;
        eng.setNode("front", front);
        eng.setNode("back", back);
        const SkeletonFrame frontLocal = withNoise(toLocal(truth, front), rng, 0.02f);
        const SkeletonFrame backLocal = withNoise(toLocal(truth, back), rng, 0.02f);
        eng.submit("front", frontLocal);
        eng.submit("back", backLocal);

        SkeletonFrame fused;
        REQUIRE(eng.fuse(t + 0.01, fused));
        REQUIRE(fused.hasBody);
        CHECK(fused.timestamp == doctest::Approx(t + 0.01));
        for (size_t i = 0; i < kJointCount; ++i) {
            const Joint j = static_cast<Joint>(i);
            REQUIRE(fused[j].state != TrackState::NotTracked);
            CHECK_FALSE(fused[j].hasRot); // sensor orientations discarded
            fusedErr += (fused[j].pos - truth[j].pos).norm();
            singleErr += (front.apply(frontLocal[j].pos) - truth[j].pos).norm();
            ++samples;
        }
    }
    REQUIRE(samples > 0);
    // Averaging two independent views should clearly beat one noisy view.
    CHECK(fusedErr / samples < 0.9 * (singleErr / samples));
}

TEST_CASE("fusion: stale node frames are excluded") {
    const double t0 = 50.0;
    const SkeletonFrame truth = standingBody(t0);
    FusionEngine eng; // default staleSeconds = 0.15
    eng.setNode("front", frontExtrinsic());
    eng.setNode("back", backExtrinsic());

    const SkeletonFrame frontLocal = toLocal(truth, frontExtrinsic());
    SkeletonFrame backLocal = toLocal(truth, backExtrinsic());
    backLocal.timestamp = t0 - 1.0; // stale by fuse time
    for (JointSample& s : backLocal.joints)
        s.pos += Vec3(5.0f, 0.0f, 0.0f); // garbage that must not contribute

    eng.submit("front", frontLocal);
    eng.submit("back", backLocal);

    SkeletonFrame fused;
    REQUIRE(eng.fuse(t0 + 0.05, fused)); // front age 0.05 fresh, back age 1.05 stale
    REQUIRE(fused.hasBody);
    for (size_t i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        CHECK((fused[j].pos - truth[j].pos).norm() < 1e-3f);
    }

    // Once everything is stale, fuse reports no body.
    SkeletonFrame none;
    CHECK_FALSE(eng.fuse(t0 + 10.0, none));
    CHECK_FALSE(none.hasBody);
}

TEST_CASE("fusion: far-side joints are occlusion-downweighted") {
    const double t0 = 20.0;
    SkeletonFrame truth = standingBody(t0);
    // Left wrist held behind the back: 0.45 m on the far side of the torso
    // plane as seen from the front sensor, near side for the back sensor.
    truth[Joint::WristL].pos = Vec3(-0.30f, 1.00f, 0.45f);

    SkeletonFrame frontLocal = toLocal(truth, frontExtrinsic());
    const SkeletonFrame backLocal = toLocal(truth, backExtrinsic());
    // The front sensor cannot actually see that wrist; simulate a bad
    // estimate (front extrinsic is a pure translation, so local x == world x).
    frontLocal[Joint::WristL].pos += Vec3(0.5f, 0.0f, 0.0f);

    auto wristError = [&](float occlusionPenalty) {
        FusionConfig cfg;
        cfg.occlusionPenalty = occlusionPenalty;
        FusionEngine eng(cfg);
        eng.setNode("front", frontExtrinsic());
        eng.setNode("back", backExtrinsic());
        eng.submit("front", frontLocal);
        eng.submit("back", backLocal);
        SkeletonFrame fused;
        REQUIRE(eng.fuse(t0 + 0.01, fused));
        return (fused[Joint::WristL].pos - truth[Joint::WristL].pos).norm();
    };

    const float errPenalized = wristError(0.3f);
    const float errUnpenalized = wristError(1.0f); // occlusion weighting disabled
    // The wrong far-side view barely moves the fused result...
    CHECK(errPenalized < 0.15f);
    // ...and clearly less than without the occlusion penalty.
    CHECK(errPenalized < 0.6f * errUnpenalized);
}

TEST_CASE("fusion: NotTracked and low-confidence samples are ignored") {
    const double t0 = 30.0;
    const SkeletonFrame truth = standingBody(t0);
    SkeletonFrame frontLocal = toLocal(truth, frontExtrinsic());
    SkeletonFrame backLocal = toLocal(truth, backExtrinsic());

    // NotTracked garbage on one node must not contribute.
    frontLocal[Joint::WristR].pos = Vec3(3.0f, 3.0f, 3.0f);
    frontLocal[Joint::WristR].state = TrackState::NotTracked;
    frontLocal[Joint::WristR].confidence = 0.0f;

    // Below-minConfidence garbage on one node must not contribute either.
    frontLocal[Joint::ElbowR].pos = Vec3(-5.0f, 0.0f, 0.0f);
    frontLocal[Joint::ElbowR].confidence = 0.01f; // default minConfidence = 0.05

    // A joint nobody tracks stays NotTracked in the fused output.
    frontLocal[Joint::WristL].state = TrackState::NotTracked;
    backLocal[Joint::WristL].state = TrackState::NotTracked;

    FusionEngine eng;
    eng.setNode("front", frontExtrinsic());
    eng.setNode("back", backExtrinsic());
    eng.submit("front", frontLocal);
    eng.submit("back", backLocal);

    SkeletonFrame fused;
    REQUIRE(eng.fuse(t0 + 0.01, fused));
    REQUIRE(fused.hasBody);

    CHECK((fused[Joint::WristR].pos - truth[Joint::WristR].pos).norm() < 1e-3f);
    CHECK(fused[Joint::WristR].state == TrackState::Tracked);
    CHECK((fused[Joint::ElbowR].pos - truth[Joint::ElbowR].pos).norm() < 1e-3f);
    CHECK(fused[Joint::WristL].state == TrackState::NotTracked);
    // The rest of the body still fuses normally.
    CHECK((fused[Joint::Hips].pos - truth[Joint::Hips].pos).norm() < 1e-3f);
}

namespace {

// Side sensor at world (2.5, 1, 0) looking along -X toward the user: local +Z
// (toward the user) maps to world -X, i.e. a -90 degree yaw.
Pose sideExtrinsic() {
    const Quat r(Eigen::AngleAxisf(-0.5f * std::numbers::pi_v<float>, Vec3::UnitY()));
    return Pose{Vec3(2.5f, 1.0f, 0.0f), r};
}

} // namespace

TEST_CASE("fusion: outlier rejection suppresses one glitching view of a joint") {
    const double t0 = 70.0;
    const SkeletonFrame truth = standingBody(t0);
    const Pose front = frontExtrinsic();
    const Pose back = backExtrinsic();
    const Pose side = sideExtrinsic();

    const SkeletonFrame frontLocal = toLocal(truth, front);
    SkeletonFrame backLocal = toLocal(truth, back);
    const SkeletonFrame sideLocal = toLocal(truth, side);
    // One node reports the right wrist a full meter off (glitch or bad
    // calibration). Rigid transforms preserve distances, so this is a 1.0 m
    // world-space error from that view.
    backLocal[Joint::WristR].pos += Vec3(1.0f, 0.0f, 0.0f);

    auto wristError = [&](bool rejection) {
        FusionConfig cfg;
        cfg.outlierRejection = rejection;
        FusionEngine eng(cfg);
        eng.setNode("front", front);
        eng.setNode("back", back);
        eng.setNode("side", side);
        eng.submit("front", frontLocal);
        eng.submit("back", backLocal);
        eng.submit("side", sideLocal);
        SkeletonFrame fused;
        REQUIRE(eng.fuse(t0 + 0.01, fused));
        REQUIRE(fused.hasBody);
        return (fused[Joint::WristR].pos - truth[Joint::WristR].pos).norm();
    };

    // With rejection on, the two agreeing views form the consensus and the
    // glitching node is downweighted to outlierWeightFactor: the fused joint
    // barely moves.
    CHECK(wristError(true) < 0.05f);
    // Without rejection the meter-off sample drags the weighted mean visibly.
    CHECK(wristError(false) > 0.15f);
}

TEST_CASE("fusion: two disagreeing nodes are both penalized (symmetric mistrust)") {
    // Property: with exactly two contributors there is no majority to decide
    // which view is wrong - each candidate's "consensus" is just the other
    // view. When they disagree beyond outlierThresholdMeters BOTH get the
    // outlier penalty (symmetric mistrust), so the fused position remains the
    // finite weighted mean BETWEEN the two views rather than snapping to
    // either one, and the fused confidence collapses to flag the conflict.
    const double t0 = 80.0;
    const SkeletonFrame truth = standingBody(t0);
    const Pose front = frontExtrinsic();
    const Pose back = backExtrinsic();

    SkeletonFrame frontLocal = toLocal(truth, front);
    SkeletonFrame backLocal = toLocal(truth, back);
    // Front extrinsic is a pure translation (local x == world x); back is a
    // 180 degree yaw (local x == world -x). Push the views 0.8 m apart.
    frontLocal[Joint::WristR].pos += Vec3(0.4f, 0.0f, 0.0f); // world +0.4 x
    backLocal[Joint::WristR].pos += Vec3(0.4f, 0.0f, 0.0f);  // world -0.4 x

    FusionEngine eng; // outlier rejection on by default
    eng.setNode("front", front);
    eng.setNode("back", back);
    eng.submit("front", frontLocal);
    eng.submit("back", backLocal);

    SkeletonFrame fused;
    REQUIRE(eng.fuse(t0 + 0.01, fused));
    REQUIRE(fused.hasBody);

    const Vec3 hi = truth[Joint::WristR].pos + Vec3(0.4f, 0.0f, 0.0f);
    const Vec3 lo = truth[Joint::WristR].pos - Vec3(0.4f, 0.0f, 0.0f);
    const Vec3 p = fused[Joint::WristR].pos;
    CHECK(std::isfinite(p.x()));
    CHECK(std::isfinite(p.y()));
    CHECK(std::isfinite(p.z()));
    // Between the two views along the axis of disagreement...
    CHECK(p.x() >= lo.x() - 1e-4f);
    CHECK(p.x() <= hi.x() + 1e-4f);
    // ...and unmoved on the axes the views agree on.
    CHECK(p.y() == doctest::Approx(truth[Joint::WristR].pos.y()).epsilon(1e-3));
    CHECK(p.z() == doctest::Approx(truth[Joint::WristR].pos.z()).epsilon(1e-3));
    // Both views were penalized: the joint's total weight (and therefore its
    // fused confidence) collapses instead of staying near 1.0.
    CHECK(fused[Joint::WristR].confidence < 0.2f);
    // Joints the nodes agree on are untouched by the rejection pass.
    CHECK((fused[Joint::Hips].pos - truth[Joint::Hips].pos).norm() < 1e-3f);
    CHECK(fused[Joint::Hips].confidence == doctest::Approx(1.0f));
}
