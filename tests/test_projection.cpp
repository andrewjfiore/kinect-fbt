#include <doctest/doctest.h>

#include "mn/projection.hpp"

#include <cmath>
#include <limits>

using namespace mn;

namespace {

// Standing body at the world origin, facing world -Z; person's right is +X.
// Mirrors the fixture used by the mapping tests so the geometry is familiar.
SkeletonFrame standingBody() {
    SkeletonFrame f;
    f.hasBody = true;
    const auto set = [&](Joint j, float x, float y, float z) {
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

} // namespace

TEST_CASE("ProjectionCorrection default is inactive and a no-op") {
    ProjectionCorrection c;
    CHECK_FALSE(c.active());

    const SkeletonFrame before = standingBody();
    SkeletonFrame after = before;
    applyProjectionCorrection(after, c);
    for (size_t i = 0; i < kJointCount; ++i) {
        CHECK(after.joints[i].pos.x() == doctest::Approx(before.joints[i].pos.x()));
        CHECK(after.joints[i].pos.y() == doctest::Approx(before.joints[i].pos.y()));
        CHECK(after.joints[i].pos.z() == doctest::Approx(before.joints[i].pos.z()));
    }
}

TEST_CASE("Single-axis flips negate exactly that world coordinate") {
    const SkeletonFrame base = standingBody();

    SUBCASE("flipX negates X only") {
        ProjectionCorrection c;
        c.flipX = true;
        CHECK(c.active());
        SkeletonFrame f = base;
        applyProjectionCorrection(f, c);
        for (size_t i = 0; i < kJointCount; ++i) {
            CHECK(f.joints[i].pos.x() == doctest::Approx(-base.joints[i].pos.x()));
            CHECK(f.joints[i].pos.y() == doctest::Approx(base.joints[i].pos.y()));
            CHECK(f.joints[i].pos.z() == doctest::Approx(base.joints[i].pos.z()));
        }
    }
    SUBCASE("flipY negates Y only") {
        ProjectionCorrection c;
        c.flipY = true;
        SkeletonFrame f = base;
        applyProjectionCorrection(f, c);
        CHECK(f[Joint::Head].pos.y() == doctest::Approx(-base[Joint::Head].pos.y()));
        CHECK(f[Joint::Head].pos.x() == doctest::Approx(base[Joint::Head].pos.x()));
    }
    SUBCASE("flipZ negates Z only") {
        ProjectionCorrection c;
        c.flipZ = true;
        SkeletonFrame f = base;
        applyProjectionCorrection(f, c);
        CHECK(f[Joint::FootL].pos.z() == doctest::Approx(-base[Joint::FootL].pos.z()));
        CHECK(f[Joint::FootL].pos.x() == doctest::Approx(base[Joint::FootL].pos.x()));
    }
}

TEST_CASE("flipX + flipZ is a 180-degree yaw that keeps the body upright") {
    // A facing-backwards rig is corrected by negating X and Z (a proper
    // rotation about the vertical axis): handedness and height are preserved.
    const SkeletonFrame base = standingBody();
    ProjectionCorrection c;
    c.flipX = true;
    c.flipZ = true;
    SkeletonFrame f = base;
    applyProjectionCorrection(f, c);

    // Head/hips heights unchanged -> still upright.
    CHECK(f[Joint::Head].pos.y() == doctest::Approx(base[Joint::Head].pos.y()));
    CHECK(f[Joint::Hips].pos.y() == doctest::Approx(base[Joint::Hips].pos.y()));
    // A foot that pointed toward -Z now points toward +Z, and its X mirrored.
    CHECK(f[Joint::FootR].pos.z() == doctest::Approx(-base[Joint::FootR].pos.z()));
    CHECK(f[Joint::FootR].pos.x() == doctest::Approx(-base[Joint::FootR].pos.x()));

    const ProjectionCheck chk = checkProjection(f);
    CHECK(chk.ok());
    CHECK(chk.uprightOk);
}

TEST_CASE("swapLR exchanges left/right joint samples") {
    const SkeletonFrame base = standingBody();
    ProjectionCorrection c;
    c.swapLR = true;
    SkeletonFrame f = base;
    applyProjectionCorrection(f, c);

    // The left wrist now holds what the right wrist held, and vice versa.
    CHECK(f[Joint::WristL].pos.x() == doctest::Approx(base[Joint::WristR].pos.x()));
    CHECK(f[Joint::WristR].pos.x() == doctest::Approx(base[Joint::WristL].pos.x()));
    CHECK(f[Joint::AnkleL].pos.x() == doctest::Approx(base[Joint::AnkleR].pos.x()));
    // Central joints are untouched.
    CHECK(f[Joint::Head].pos.x() == doctest::Approx(base[Joint::Head].pos.x()));
    CHECK(f[Joint::Spine].pos.y() == doctest::Approx(base[Joint::Spine].pos.y()));
}

TEST_CASE("flipX + swapLR un-mirrors a sensor-mirrored skeleton") {
    // Kinect mirror mode reflects the body across X and swaps the labels.
    // Undo it with the same two operations; the result matches the original.
    const SkeletonFrame original = standingBody();

    // Simulate the sensor's mirroring: reflect X and relabel L<->R.
    SkeletonFrame mirrored = original;
    for (JointSample& s : mirrored.joints)
        s.pos.x() = -s.pos.x();
    ProjectionCorrection relabel;
    relabel.swapLR = true;
    applyProjectionCorrection(mirrored, relabel);

    // Now correct it back.
    ProjectionCorrection fix;
    fix.flipX = true;
    fix.swapLR = true;
    applyProjectionCorrection(mirrored, fix);

    for (size_t i = 0; i < kJointCount; ++i) {
        CHECK(mirrored.joints[i].pos.x() == doctest::Approx(original.joints[i].pos.x()));
        CHECK(mirrored.joints[i].pos.y() == doctest::Approx(original.joints[i].pos.y()));
        CHECK(mirrored.joints[i].pos.z() == doctest::Approx(original.joints[i].pos.z()));
    }
}

TEST_CASE("checkProjection passes a healthy standing body") {
    const ProjectionCheck chk = checkProjection(standingBody());
    CHECK(chk.evaluated);
    CHECK(chk.ok());
    CHECK(chk.finite);
    CHECK(chk.inBounds);
    CHECK(chk.bonesPlausible);
    CHECK(chk.uprightOk);
    CHECK(chk.nonFiniteJoints == 0);
    CHECK(chk.headAboveHipsMeters > 0.0f);
    CHECK(chk.summary() == "ok");
}

TEST_CASE("checkProjection reports no body for an empty frame") {
    SkeletonFrame empty; // hasBody = false
    const ProjectionCheck chk = checkProjection(empty);
    CHECK_FALSE(chk.evaluated);
    CHECK(chk.ok()); // an unevaluated frame is not a failure
    CHECK(chk.summary() == "no body");
}

TEST_CASE("checkProjection flags non-finite joints") {
    SkeletonFrame f = standingBody();
    f[Joint::WristR].pos.x() = std::numeric_limits<float>::quiet_NaN();
    f[Joint::AnkleL].pos.y() = std::numeric_limits<float>::infinity();
    const ProjectionCheck chk = checkProjection(f);
    CHECK_FALSE(chk.ok());
    CHECK_FALSE(chk.finite);
    CHECK(chk.nonFiniteJoints == 2);
}

TEST_CASE("checkProjection flags a joint flung out of bounds") {
    SkeletonFrame f = standingBody();
    f[Joint::WristR].pos = Vec3(50.0f, 0.0f, 0.0f); // miscalibration blow-up
    const ProjectionCheck chk = checkProjection(f);
    CHECK_FALSE(chk.ok());
    CHECK_FALSE(chk.inBounds);
    CHECK(chk.outOfBoundsJoints >= 1);
    CHECK(chk.worstRadiusMeters >= 50.0f);
    // The stretched forearm also reads as an implausible bone.
    CHECK_FALSE(chk.bonesPlausible);
}

TEST_CASE("checkProjection detects an upside-down (flipped Y) skeleton") {
    SkeletonFrame f = standingBody();
    ProjectionCorrection flipY;
    flipY.flipY = true;
    applyProjectionCorrection(f, flipY);
    const ProjectionCheck chk = checkProjection(f);
    CHECK_FALSE(chk.uprightOk);
    CHECK_FALSE(chk.ok());
    CHECK(chk.headAboveHipsMeters < 0.0f);
}

TEST_CASE("checkProjection tolerates a body that is only partially tracked") {
    SkeletonFrame f = standingBody();
    // Drop the arms: untracked joints must not count against the check.
    for (Joint j : {Joint::ShoulderL, Joint::ElbowL, Joint::WristL, Joint::ShoulderR,
                    Joint::ElbowR, Joint::WristR}) {
        f[j] = JointSample{}; // state = NotTracked
    }
    const ProjectionCheck chk = checkProjection(f);
    CHECK(chk.ok());
    CHECK(chk.trackedJoints == static_cast<int>(kJointCount) - 6);
}
