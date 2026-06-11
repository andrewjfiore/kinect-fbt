#pragma once
// Math primitives. Conventions (see docs/DESIGN.md):
//  - All positions in meters.
//  - World frame: right-handed, +Y up, -Z "forward" (OpenVR convention).
//  - Node-local frame: right-handed, +Y up, +Z pointing from the sensor toward
//    the tracked user (Kinect camera-space convention).
//  - Orientations: a tracker/device's local axes are +X right, +Y up, -Z forward.

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace mn {

using Vec3 = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Mat3 = Eigen::Matrix3f;

// Rigid transform (SE(3)). Maps points: out = rot * p + pos.
struct Pose {
    Vec3 pos{Vec3::Zero()};
    Quat rot{Quat::Identity()};

    Vec3 apply(const Vec3& p) const { return rot * p + pos; }
    Quat applyRot(const Quat& q) const { return (rot * q).normalized(); }

    Pose inverse() const {
        const Quat ri = rot.conjugate();
        return Pose{ri * (-pos), ri};
    }

    // this ∘ other: applies `other` first, then `this`.
    Pose compose(const Pose& other) const {
        return Pose{apply(other.pos), (rot * other.rot).normalized()};
    }

    static Pose identity() { return {}; }
};

// Normalize, falling back to `fallback` for near-zero vectors.
inline Vec3 safeNormalized(const Vec3& v, const Vec3& fallback = Vec3::UnitY()) {
    const float n = v.norm();
    return (n > 1e-6f) ? Vec3(v / n) : fallback;
}

// Rotation whose local -Z axis points along `forward` and whose +Y is as close
// to `up` as possible (Gram-Schmidt). Both inputs need not be normalized.
inline Quat quatFromForwardUp(const Vec3& forward, const Vec3& up = Vec3::UnitY()) {
    const Vec3 f = safeNormalized(forward, -Vec3::UnitZ());
    Vec3 u = up - f * up.dot(f);
    u = safeNormalized(u, (std::abs(f.y()) > 0.99f) ? Vec3::UnitZ() : Vec3::UnitY());
    const Vec3 r = u.cross(-f).normalized(); // right = up x backward... see below
    Mat3 m;
    // Columns are the local axes expressed in the parent frame: X=right, Y=up, Z=backward.
    m.col(0) = r;
    m.col(1) = u;
    m.col(2) = -f;
    return Quat(m).normalized();
}

// Smallest-angle quaternion delta magnitude in radians.
inline float quatAngle(const Quat& a, const Quat& b) {
    const float d = std::min(1.0f, std::abs(a.dot(b)));
    return 2.0f * std::acos(d);
}

} // namespace mn
