#pragma once
// Maps a fused world-frame skeleton onto a configurable set of virtual
// trackers. Joint orientations from sensors are noisy, so tracker orientations
// are DERIVED from fused joint positions + the skeletal model:
//   waist: right = hipR-hipL, up = spine-hips
//   chest: right = shoulderR-shoulderL, up = neck-chest
//   feet:  forward = horizontal(foot-ankle), up = world up
//   knees: forward = knee->ankle projected, up along thigh
//   elbows: forward along forearm, up along upper arm
// Velocities are finite-differenced and EMA-smoothed so SteamVR/consumers can
// predict over pipeline latency.

#include "mn/skeleton.hpp"

#include <optional>
#include <string_view>
#include <vector>

namespace mn {

enum class TrackerRole : uint8_t {
    Waist = 0,
    LeftFoot,
    RightFoot,
    Chest,
    LeftKnee,
    RightKnee,
    LeftElbow,
    RightElbow,
    Head, // reference for VRChat OSC alignment; not a SteamVR tracker
    Count
};

// "waist", "left_foot", "right_foot", "chest", "left_knee", "right_knee",
// "left_elbow", "right_elbow", "head"
const char* trackerRoleName(TrackerRole r);
std::optional<TrackerRole> trackerRoleFromName(std::string_view name);

struct TrackerPose {
    TrackerRole role = TrackerRole::Waist;
    bool valid = false;
    Pose pose;                          // world frame
    Vec3 velocity{Vec3::Zero()};        // m/s, world frame
    Vec3 angularVelocity{Vec3::Zero()}; // rad/s, world frame
};

struct MappingConfig {
    std::vector<TrackerRole> roles = {TrackerRole::Waist, TrackerRole::LeftFoot,
                                      TrackerRole::RightFoot};
    bool emitHead = true;        // append a Head reference pose
    float velocitySmooth = 0.5f; // EMA alpha for velocity estimates
};

class TrackerMapper {
public:
    explicit TrackerMapper(MappingConfig cfg = {});

    // One TrackerPose per configured role (+ Head when emitHead), in config
    // order. Roles whose source joints are not tracked come back valid=false.
    std::vector<TrackerPose> map(const SkeletonFrame& world, double timestamp);

    void reset();

private:
    MappingConfig cfg_;
    struct Prev {
        bool has = false;
        double t = 0.0;
        Pose pose;
        Vec3 vel{Vec3::Zero()};
        Vec3 angVel{Vec3::Zero()};
    };
    std::vector<Prev> prev_;
};

} // namespace mn
