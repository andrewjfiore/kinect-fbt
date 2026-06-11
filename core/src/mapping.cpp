// Tracker mapping: derives tracker poses (position, orientation, velocities)
// from a fused world-frame skeleton. Sensor joint orientations are noisy, so
// tracker orientations are rebuilt from joint positions using the recipes
// documented in mn/mapping.hpp.

#include "mn/mapping.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace mn {

namespace {

constexpr float kMinSourceConfidence = 0.05f;

constexpr std::array<const char*, static_cast<size_t>(TrackerRole::Count)> kRoleNames = {
    "waist",     "left_foot",  "right_foot",  "chest", "left_knee",
    "right_knee", "left_elbow", "right_elbow", "head"};

bool usableJoint(const JointSample& s) {
    return s.state != TrackState::NotTracked && s.confidence >= kMinSourceConfidence;
}

Joint sourceJoint(TrackerRole r) {
    switch (r) {
    case TrackerRole::Waist:
        return Joint::Hips;
    case TrackerRole::LeftFoot:
        return Joint::AnkleL;
    case TrackerRole::RightFoot:
        return Joint::AnkleR;
    case TrackerRole::Chest:
        return Joint::Chest;
    case TrackerRole::LeftKnee:
        return Joint::KneeL;
    case TrackerRole::RightKnee:
        return Joint::KneeR;
    case TrackerRole::LeftElbow:
        return Joint::ElbowL;
    case TrackerRole::RightElbow:
        return Joint::ElbowR;
    case TrackerRole::Head:
        return Joint::Head;
    default:
        return Joint::Hips;
    }
}

// Body facing direction. With device axes +X right / +Y up / -Z forward,
// forward = up x right.
Vec3 torsoForward(const SkeletonFrame& w) {
    const Vec3 right = w[Joint::HipR].pos - w[Joint::HipL].pos;
    const Vec3 up = w[Joint::Spine].pos - w[Joint::Hips].pos;
    return up.cross(right);
}

Quat footOrientation(const SkeletonFrame& w, bool left) {
    const Joint ankle = left ? Joint::AnkleL : Joint::AnkleR;
    const Joint foot = left ? Joint::FootL : Joint::FootR;
    Vec3 fwd = Vec3::Zero();
    if (usableJoint(w[foot]) && usableJoint(w[ankle]))
        fwd = w[foot].pos - w[ankle].pos;
    fwd.y() = 0.0f; // horizontal foot direction
    if (fwd.squaredNorm() < 1e-8f) {
        fwd = torsoForward(w); // fall back to body facing
        fwd.y() = 0.0f;
    }
    return quatFromForwardUp(fwd, Vec3::UnitY());
}

Quat kneeOrientation(const SkeletonFrame& w, bool left) {
    const Joint hip = left ? Joint::HipL : Joint::HipR;
    const Joint knee = left ? Joint::KneeL : Joint::KneeR;
    const Joint ankle = left ? Joint::AnkleL : Joint::AnkleR;
    const Vec3 up = w[hip].pos - w[knee].pos; // up along the thigh
    const Vec3 upn = safeNormalized(up);
    Vec3 fwd = w[ankle].pos - w[knee].pos; // knee -> ankle ...
    fwd -= upn * fwd.dot(upn);             // ... projected off the thigh axis
    if (fwd.squaredNorm() < 1e-8f) {
        fwd = torsoForward(w); // straight leg: fall back to body facing
        fwd -= upn * fwd.dot(upn);
    }
    return quatFromForwardUp(fwd, up);
}

Quat elbowOrientation(const SkeletonFrame& w, bool left) {
    const Joint shoulder = left ? Joint::ShoulderL : Joint::ShoulderR;
    const Joint elbow = left ? Joint::ElbowL : Joint::ElbowR;
    const Joint wrist = left ? Joint::WristL : Joint::WristR;
    const Vec3 fwd = w[wrist].pos - w[elbow].pos;   // forward along the forearm
    const Vec3 up = w[shoulder].pos - w[elbow].pos; // up along the upper arm
    return quatFromForwardUp(fwd, up);
}

Quat orientationFor(TrackerRole r, const SkeletonFrame& w) {
    switch (r) {
    case TrackerRole::Waist: {
        const Vec3 up = w[Joint::Spine].pos - w[Joint::Hips].pos;
        const Vec3 right = w[Joint::HipR].pos - w[Joint::HipL].pos;
        return quatFromForwardUp(up.cross(right), up);
    }
    case TrackerRole::Chest: {
        const Vec3 up = w[Joint::Neck].pos - w[Joint::Chest].pos;
        const Vec3 right = w[Joint::ShoulderR].pos - w[Joint::ShoulderL].pos;
        return quatFromForwardUp(up.cross(right), up);
    }
    case TrackerRole::LeftFoot:
        return footOrientation(w, true);
    case TrackerRole::RightFoot:
        return footOrientation(w, false);
    case TrackerRole::LeftKnee:
        return kneeOrientation(w, true);
    case TrackerRole::RightKnee:
        return kneeOrientation(w, false);
    case TrackerRole::LeftElbow:
        return elbowOrientation(w, true);
    case TrackerRole::RightElbow:
        return elbowOrientation(w, false);
    case TrackerRole::Head: {
        const Vec3 up = w[Joint::Head].pos - w[Joint::Neck].pos;
        const Vec3 right = w[Joint::ShoulderR].pos - w[Joint::ShoulderL].pos;
        return quatFromForwardUp(up.cross(right), up);
    }
    default:
        return Quat::Identity();
    }
}

} // namespace

const char* trackerRoleName(TrackerRole r) {
    const auto i = static_cast<size_t>(r);
    return i < kRoleNames.size() ? kRoleNames[i] : "invalid";
}

std::optional<TrackerRole> trackerRoleFromName(std::string_view name) {
    for (size_t i = 0; i < kRoleNames.size(); ++i)
        if (name == kRoleNames[i])
            return static_cast<TrackerRole>(i);
    return std::nullopt;
}

TrackerMapper::TrackerMapper(MappingConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.emitHead &&
        std::find(cfg_.roles.begin(), cfg_.roles.end(), TrackerRole::Head) == cfg_.roles.end())
        cfg_.roles.push_back(TrackerRole::Head);
    prev_.resize(cfg_.roles.size());
}

std::vector<TrackerPose> TrackerMapper::map(const SkeletonFrame& world, double timestamp) {
    std::vector<TrackerPose> out(cfg_.roles.size());
    for (size_t i = 0; i < cfg_.roles.size(); ++i) {
        const TrackerRole role = cfg_.roles[i];
        TrackerPose& tp = out[i];
        tp.role = role;

        const JointSample& src = world[sourceJoint(role)];
        Prev& pv = prev_[i];
        if (!world.hasBody || src.state == TrackState::NotTracked ||
            src.confidence < kMinSourceConfidence) {
            tp.valid = false;
            tp.pose.pos = src.pos; // best effort; consumers must check valid
            pv.has = false;        // restart velocity estimation after a dropout
            continue;
        }

        tp.valid = true;
        tp.pose.pos = src.pos;
        tp.pose.rot = orientationFor(role, world);

        Vec3 vel = Vec3::Zero();
        Vec3 angVel = Vec3::Zero();
        if (pv.has) {
            const double dt = timestamp - pv.t;
            if (dt > 0.0) {
                const float fdt = static_cast<float>(dt);
                const Vec3 rawVel = (tp.pose.pos - pv.pose.pos) / fdt;

                Quat dq = tp.pose.rot * pv.pose.rot.conjugate();
                if (dq.w() < 0.0f) // shortest-arc delta
                    dq = Quat(-dq.w(), -dq.x(), -dq.y(), -dq.z());
                dq.normalize();
                Vec3 rawAng = Vec3::Zero();
                const Vec3 axisPart = dq.vec();
                const float s = axisPart.norm();
                if (s > 1e-8f) {
                    const float angle = 2.0f * std::atan2(s, dq.w());
                    rawAng = (axisPart / s) * (angle / fdt);
                }

                const float a = std::clamp(cfg_.velocitySmooth, 0.0f, 1.0f);
                vel = pv.vel + a * (rawVel - pv.vel);
                angVel = pv.angVel + a * (rawAng - pv.angVel);
            }
            // dt <= 0 (duplicate or out-of-order tick): report zero per contract.
        }
        tp.velocity = vel;
        tp.angularVelocity = angVel;

        pv.has = true;
        pv.t = timestamp;
        pv.pose = tp.pose;
        pv.vel = vel;
        pv.angVel = angVel;
    }
    return out;
}

void TrackerMapper::reset() {
    for (Prev& p : prev_)
        p = Prev{};
}

} // namespace mn
