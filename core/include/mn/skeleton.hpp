#pragma once
// Common joint schema. Every capture backend (Kinect v1's 20 joints, v2's 25,
// markerless models) normalizes to this FBT-relevant subset.

#include "mn/math.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace mn {

enum class Joint : uint8_t {
    Head = 0,
    Neck,
    Chest,
    Spine,
    Hips,
    ShoulderL,
    ElbowL,
    WristL,
    ShoulderR,
    ElbowR,
    WristR,
    HipL,
    KneeL,
    AnkleL,
    FootL,
    HipR,
    KneeR,
    AnkleR,
    FootR,
    Count
};

inline constexpr size_t kJointCount = static_cast<size_t>(Joint::Count);

// Lowercase snake_case names: "head", "shoulder_l", ... Used in config/JSONL.
const char* jointName(Joint j);
std::optional<Joint> jointFromName(std::string_view name);

// Parent joint in the kinematic tree. Hips is the root (returns Hips).
Joint jointParent(Joint j);

enum class TrackState : uint8_t { NotTracked = 0, Inferred = 1, Tracked = 2 };

struct JointSample {
    Vec3 pos{Vec3::Zero()};
    Quat rot{Quat::Identity()};
    float confidence = 0.0f; // [0,1]
    TrackState state = TrackState::NotTracked;
    bool hasRot = false; // sensor provided an orientation (we mostly ignore them)
};

struct SkeletonFrame {
    double timestamp = 0.0; // mn::nowSeconds() at emission
    bool hasBody = false;
    std::array<JointSample, kJointCount> joints{};

    JointSample& operator[](Joint j) { return joints[static_cast<size_t>(j)]; }
    const JointSample& operator[](Joint j) const { return joints[static_cast<size_t>(j)]; }
};

} // namespace mn
