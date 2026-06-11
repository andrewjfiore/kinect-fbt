#pragma once
// Wire protocol between the core app (endpoints/openvr_bridge) and the
// SteamVR driver (driver/openvr). Localhost UDP, little-endian, packed.
// Deliberately self-contained: the driver includes only this header (plus
// openvr_driver.h) so it stays dependency-light.

#include <cstddef>
#include <cstdint>

namespace mn::wire {

inline constexpr uint32_t kMagic = 0x4D4E5450; // "MNTP"
inline constexpr uint16_t kVersion = 1;
inline constexpr uint16_t kDefaultPort = 24190;
inline constexpr size_t kMaxTrackers = 16;

// Mirrors mn::TrackerRole (static_assert'd in the bridge endpoint).
enum class Role : uint8_t {
    Waist = 0,
    LeftFoot,
    RightFoot,
    Chest,
    LeftKnee,
    RightKnee,
    LeftElbow,
    RightElbow,
    Head, // never sent to the driver
    Count
};

// Stable per-role driver serials.
inline const char* roleSerial(Role r) {
    switch (r) {
    case Role::Waist: return "MN-WAIST";
    case Role::LeftFoot: return "MN-LFOOT";
    case Role::RightFoot: return "MN-RFOOT";
    case Role::Chest: return "MN-CHEST";
    case Role::LeftKnee: return "MN-LKNEE";
    case Role::RightKnee: return "MN-RKNEE";
    case Role::LeftElbow: return "MN-LELBOW";
    case Role::RightElbow: return "MN-RELBOW";
    case Role::Head: return "MN-HEAD";
    default: return "MN-UNKNOWN";
    }
}

#pragma pack(push, 1)
struct WirePose {
    uint8_t role = 0;  // Role
    uint8_t valid = 0; // 0/1
    float px = 0, py = 0, pz = 0;     // meters, world frame (+Y up, -Z fwd)
    float qw = 1, qx = 0, qy = 0, qz = 0;
    float vx = 0, vy = 0, vz = 0;     // m/s
    float wx = 0, wy = 0, wz = 0;     // rad/s
};

struct WirePacket {
    uint32_t magic = kMagic;
    uint16_t version = kVersion;
    uint16_t count = 0;     // number of valid entries in poses[]; 0 = heartbeat
    double timestamp = 0.0; // sender monotonic seconds
    WirePose poses[kMaxTrackers]{};
};
#pragma pack(pop)

static_assert(sizeof(WirePose) == 2 + 13 * sizeof(float), "WirePose must be packed");
static_assert(sizeof(WirePacket) == 8 + sizeof(double) + kMaxTrackers * sizeof(WirePose),
              "WirePacket must be packed");

// Serialized size when sending only `count` poses.
inline size_t packetBytes(uint16_t count) {
    return 8 + sizeof(double) + static_cast<size_t>(count) * sizeof(WirePose);
}

} // namespace mn::wire
