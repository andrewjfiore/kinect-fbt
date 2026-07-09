#include "mn/projection.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace mn {

namespace {

// Left/right joint pairs exchanged by swapLR. Central joints (head, neck,
// chest, spine, hips) have no mirror partner and are left in place.
constexpr std::array<std::pair<Joint, Joint>, 7> kLrPairs = {{
    {Joint::ShoulderL, Joint::ShoulderR},
    {Joint::ElbowL, Joint::ElbowR},
    {Joint::WristL, Joint::WristR},
    {Joint::HipL, Joint::HipR},
    {Joint::KneeL, Joint::KneeR},
    {Joint::AnkleL, Joint::AnkleR},
    {Joint::FootL, Joint::FootR},
}};

bool finiteVec(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

} // namespace

void applyProjectionCorrection(SkeletonFrame& world, const ProjectionCorrection& c) {
    if (!c.active())
        return;

    // Relabel first: swapping the whole sample keeps confidence/state with the
    // limb they describe. Order is irrelevant -- the sign flip below is applied
    // uniformly to every slot -- but doing the swap first reads cleanly.
    if (c.swapLR) {
        for (const auto& pair : kLrPairs)
            std::swap(world[pair.first], world[pair.second]);
    }

    if (c.flipX || c.flipY || c.flipZ) {
        const float sx = c.flipX ? -1.0f : 1.0f;
        const float sy = c.flipY ? -1.0f : 1.0f;
        const float sz = c.flipZ ? -1.0f : 1.0f;
        for (auto& j : world.joints) {
            j.pos.x() *= sx;
            j.pos.y() *= sy;
            j.pos.z() *= sz;
        }
    }
}

std::string ProjectionCheck::summary() const {
    if (!evaluated)
        return "no body";
    if (ok())
        return "ok";
    std::string s;
    const auto add = [&](const std::string& part) {
        if (!s.empty())
            s += ", ";
        s += part;
    };
    if (!finite)
        add(std::to_string(nonFiniteJoints) + " non-finite joint(s)");
    if (!inBounds)
        add(std::to_string(outOfBoundsJoints) + " joint(s) out of bounds");
    if (!bonesPlausible)
        add(std::to_string(implausibleBones) + " implausible bone(s)");
    if (!uprightOk)
        add("skeleton inverted (head below hips)");
    return s;
}

ProjectionCheck checkProjection(const SkeletonFrame& world, const ProjectionCheckParams& p) {
    ProjectionCheck r;
    if (!world.hasBody)
        return r; // evaluated stays false: nothing to judge
    r.evaluated = true;

    const auto tracked = [&](Joint j) { return world[j].state != TrackState::NotTracked; };

    // Per-joint: finiteness + distance from the world origin.
    for (size_t i = 0; i < kJointCount; ++i) {
        const JointSample& s = world.joints[i];
        if (s.state == TrackState::NotTracked)
            continue;
        ++r.trackedJoints;
        if (!finiteVec(s.pos)) {
            ++r.nonFiniteJoints;
            continue; // a non-finite joint can't be bounds-checked
        }
        const float radius = s.pos.norm();
        if (radius > r.worstRadiusMeters)
            r.worstRadiusMeters = radius;
        if (radius > p.maxRadiusMeters)
            ++r.outOfBoundsJoints;
    }

    // Bone lengths, over segments whose endpoints are both tracked and finite.
    for (size_t i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        if (j == Joint::Hips)
            continue; // root: no parent bone
        const Joint par = jointParent(j);
        if (!tracked(j) || !tracked(par))
            continue;
        const Vec3& a = world[j].pos;
        const Vec3& b = world[par].pos;
        if (!finiteVec(a) || !finiteVec(b))
            continue;
        const float len = (a - b).norm();
        float err = 0.0f;
        if (len < p.minBoneMeters)
            err = p.minBoneMeters - len;
        else if (len > p.maxBoneMeters)
            err = len - p.maxBoneMeters;
        if (err > 0.0f) {
            ++r.implausibleBones;
            if (err > r.worstBoneErrorMeters)
                r.worstBoneErrorMeters = err;
        }
    }

    // Upright heuristic: with a normal standing rig the head sits above the
    // hips. A clearly negative gap means the vertical axis is flipped.
    if (tracked(Joint::Head) && tracked(Joint::Hips)) {
        const float dy = world[Joint::Head].pos.y() - world[Joint::Hips].pos.y();
        if (std::isfinite(dy))
            r.headAboveHipsMeters = dy;
    }

    r.finite = (r.nonFiniteJoints == 0);
    r.inBounds = (r.outOfBoundsJoints == 0);
    r.bonesPlausible = (r.implausibleBones == 0);
    r.uprightOk = (r.headAboveHipsMeters >= p.minHeadAboveHips);
    return r;
}

} // namespace mn
