#include <doctest/doctest.h>

#include "mn/mapping.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

using namespace mn;

namespace {

// Standing body at the world origin, facing world -Z. Person's right is +X.
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

SkeletonFrame shiftedX(const SkeletonFrame& f, float dx) {
    SkeletonFrame out = f;
    for (JointSample& s : out.joints)
        s.pos.x() += dx;
    return out;
}

float angleToWorldMinusZ(const Quat& q) {
    const Vec3 fwd = q * Vec3(0.0f, 0.0f, -1.0f);
    const float cosA = std::clamp(fwd.dot(Vec3(0.0f, 0.0f, -1.0f)), -1.0f, 1.0f);
    return std::acos(cosA) * 180.0f / std::numbers::pi_v<float>;
}

} // namespace

TEST_CASE("mapping: tracker role names round-trip") {
    for (size_t i = 0; i < static_cast<size_t>(TrackerRole::Count); ++i) {
        const TrackerRole r = static_cast<TrackerRole>(i);
        const auto parsed = trackerRoleFromName(trackerRoleName(r));
        REQUIRE(parsed.has_value());
        CHECK(*parsed == r);
    }
    CHECK(std::string(trackerRoleName(TrackerRole::Waist)) == "waist");
    CHECK(std::string(trackerRoleName(TrackerRole::LeftFoot)) == "left_foot");
    CHECK(std::string(trackerRoleName(TrackerRole::RightElbow)) == "right_elbow");
    CHECK(std::string(trackerRoleName(TrackerRole::Head)) == "head");
    CHECK_FALSE(trackerRoleFromName("left_pinky").has_value());
    CHECK_FALSE(trackerRoleFromName("").has_value());
}

TEST_CASE("mapping: forward-facing skeleton yields forward-facing waist") {
    const SkeletonFrame world = standingBody(0.0);
    TrackerMapper mapper; // default roles: waist + feet, head appended
    const auto poses = mapper.map(world, 0.0);

    REQUIRE(poses.size() == 4);
    CHECK(poses[0].role == TrackerRole::Waist);
    CHECK(poses[1].role == TrackerRole::LeftFoot);
    CHECK(poses[2].role == TrackerRole::RightFoot);
    CHECK(poses[3].role == TrackerRole::Head);

    REQUIRE(poses[0].valid);
    CHECK((poses[0].pose.pos - world[Joint::Hips].pos).norm() < 1e-5f);
    // Tracker -Z (forward) within 15 degrees of world -Z.
    CHECK(angleToWorldMinusZ(poses[0].pose.rot) < 15.0f);

    // Feet sit on the ankles and also point roughly forward (toes at -Z).
    REQUIRE(poses[1].valid);
    CHECK((poses[1].pose.pos - world[Joint::AnkleL].pos).norm() < 1e-5f);
    CHECK(angleToWorldMinusZ(poses[1].pose.rot) < 15.0f);

    // First frame: velocities are zero.
    CHECK(poses[0].velocity.norm() < 1e-6f);
    CHECK(poses[0].angularVelocity.norm() < 1e-6f);
}

TEST_CASE("mapping: configured roles map in config order") {
    MappingConfig cfg;
    cfg.roles = {TrackerRole::Chest, TrackerRole::LeftElbow, TrackerRole::RightKnee};
    cfg.emitHead = false;
    TrackerMapper mapper(cfg);

    const SkeletonFrame world = standingBody(0.0);
    const auto poses = mapper.map(world, 0.0);
    REQUIRE(poses.size() == 3);
    CHECK(poses[0].role == TrackerRole::Chest);
    CHECK(poses[1].role == TrackerRole::LeftElbow);
    CHECK(poses[2].role == TrackerRole::RightKnee);
    for (const TrackerPose& tp : poses)
        CHECK(tp.valid);
    CHECK((poses[0].pose.pos - world[Joint::Chest].pos).norm() < 1e-5f);
    CHECK((poses[1].pose.pos - world[Joint::ElbowL].pos).norm() < 1e-5f);
    CHECK((poses[2].pose.pos - world[Joint::KneeR].pos).norm() < 1e-5f);
}

TEST_CASE("mapping: waist velocity follows motion") {
    TrackerMapper mapper; // default velocitySmooth = 0.5
    const SkeletonFrame base = standingBody(0.0);
    const double dt = 0.01;

    std::vector<TrackerPose> poses;
    for (int i = 0; i < 10; ++i) {
        const double t = static_cast<double>(i) * dt;
        SkeletonFrame w = shiftedX(base, static_cast<float>(t)); // 1 m/s along +X
        w.timestamp = t;
        poses = mapper.map(w, t);
    }
    REQUIRE(!poses.empty());
    REQUIRE(poses[0].valid);
    // After a few frames the EMA has converged toward +1 m/s along X.
    CHECK(poses[0].velocity.x() > 0.5f);
    CHECK(std::abs(poses[0].velocity.y()) < 0.2f);
    CHECK(std::abs(poses[0].velocity.z()) < 0.2f);
    // Orientation is constant, so angular velocity stays near zero.
    CHECK(poses[0].angularVelocity.norm() < 0.5f);

    // reset() clears history: the next frame reports zero velocity again.
    mapper.reset();
    SkeletonFrame w = shiftedX(base, 1.0f);
    w.timestamp = 1.0;
    poses = mapper.map(w, 1.0);
    REQUIRE(poses[0].valid);
    CHECK(poses[0].velocity.norm() < 1e-6f);
    CHECK(poses[0].angularVelocity.norm() < 1e-6f);
}

TEST_CASE("mapping: untracked source joints invalidate trackers") {
    SkeletonFrame w = standingBody(0.0);
    w[Joint::Hips].state = TrackState::NotTracked;
    w[Joint::AnkleL].confidence = 0.0f; // below the 0.05 validity floor

    TrackerMapper mapper;
    const auto poses = mapper.map(w, 0.0);
    REQUIRE(poses.size() == 4);
    CHECK_FALSE(poses[0].valid); // waist: hips not tracked
    CHECK_FALSE(poses[1].valid); // left foot: ankle confidence too low
    CHECK(poses[2].valid);       // right foot unaffected
    CHECK(poses[3].valid);       // head unaffected

    // No body at all: every tracker is invalid.
    SkeletonFrame empty;
    const auto p2 = TrackerMapper().map(empty, 0.0);
    REQUIRE(p2.size() == 4);
    for (const TrackerPose& tp : p2)
        CHECK_FALSE(tp.valid);
}
