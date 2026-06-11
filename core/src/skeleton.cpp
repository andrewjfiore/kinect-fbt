#include "mn/skeleton.hpp"

#include <array>
#include <cstring>

namespace mn {

namespace {
constexpr std::array<const char*, kJointCount> kNames = {
    "head",     "neck",       "chest",   "spine",   "hips",    "shoulder_l", "elbow_l",
    "wrist_l",  "shoulder_r", "elbow_r", "wrist_r", "hip_l",   "knee_l",     "ankle_l",
    "foot_l",   "hip_r",      "knee_r",  "ankle_r", "foot_r"};

constexpr std::array<Joint, kJointCount> kParents = {
    /*Head*/ Joint::Neck,
    /*Neck*/ Joint::Chest,
    /*Chest*/ Joint::Spine,
    /*Spine*/ Joint::Hips,
    /*Hips*/ Joint::Hips, // root
    /*ShoulderL*/ Joint::Chest,
    /*ElbowL*/ Joint::ShoulderL,
    /*WristL*/ Joint::ElbowL,
    /*ShoulderR*/ Joint::Chest,
    /*ElbowR*/ Joint::ShoulderR,
    /*WristR*/ Joint::ElbowR,
    /*HipL*/ Joint::Hips,
    /*KneeL*/ Joint::HipL,
    /*AnkleL*/ Joint::KneeL,
    /*FootL*/ Joint::AnkleL,
    /*HipR*/ Joint::Hips,
    /*KneeR*/ Joint::HipR,
    /*AnkleR*/ Joint::KneeR,
    /*FootR*/ Joint::AnkleR};
} // namespace

const char* jointName(Joint j) {
    const auto i = static_cast<size_t>(j);
    return i < kJointCount ? kNames[i] : "invalid";
}

std::optional<Joint> jointFromName(std::string_view name) {
    for (size_t i = 0; i < kJointCount; ++i)
        if (name == kNames[i])
            return static_cast<Joint>(i);
    return std::nullopt;
}

Joint jointParent(Joint j) {
    const auto i = static_cast<size_t>(j);
    return i < kJointCount ? kParents[i] : Joint::Hips;
}

} // namespace mn
