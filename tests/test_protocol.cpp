// Wire-protocol layout tests: the openvr_bridge endpoint and the SteamVR
// driver are built separately (the driver may even be a different compiler),
// so the packed layout in mn/protocol.hpp must be byte-exact.

#include "mn/protocol.hpp"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using namespace mn::wire;

TEST_CASE("protocol: struct sizes are packed") {
    // Compile-time guarantees (mirrors the static_asserts in protocol.hpp)...
    static_assert(sizeof(WirePose) == 54);
    static_assert(sizeof(WirePacket) == 16 + kMaxTrackers * 54);
    // ...and the same as runtime checks so a failure shows up in test output.
    CHECK(sizeof(WirePose) == 2 + 13 * sizeof(float));
    CHECK(sizeof(WirePose) == 54);
    CHECK(sizeof(WirePacket) == 8 + sizeof(double) + kMaxTrackers * sizeof(WirePose));
    CHECK(sizeof(WirePacket) == 880);
}

TEST_CASE("protocol: WirePose field offsets") {
    CHECK(offsetof(WirePose, role) == 0);
    CHECK(offsetof(WirePose, valid) == 1);
    CHECK(offsetof(WirePose, px) == 2);
    CHECK(offsetof(WirePose, py) == 6);
    CHECK(offsetof(WirePose, pz) == 10);
    CHECK(offsetof(WirePose, qw) == 14);
    CHECK(offsetof(WirePose, qx) == 18);
    CHECK(offsetof(WirePose, qy) == 22);
    CHECK(offsetof(WirePose, qz) == 26);
    CHECK(offsetof(WirePose, vx) == 30);
    CHECK(offsetof(WirePose, vy) == 34);
    CHECK(offsetof(WirePose, vz) == 38);
    CHECK(offsetof(WirePose, wx) == 42);
    CHECK(offsetof(WirePose, wy) == 46);
    CHECK(offsetof(WirePose, wz) == 50);
}

TEST_CASE("protocol: WirePacket field offsets") {
    CHECK(offsetof(WirePacket, magic) == 0);
    CHECK(offsetof(WirePacket, version) == 4);
    CHECK(offsetof(WirePacket, count) == 6);
    CHECK(offsetof(WirePacket, timestamp) == 8);
    CHECK(offsetof(WirePacket, poses) == 16);
}

TEST_CASE("protocol: packetBytes math") {
    CHECK(packetBytes(0) == 16);                              // heartbeat: header only
    CHECK(packetBytes(0) == offsetof(WirePacket, poses));     // header == poses offset
    CHECK(packetBytes(1) == 16 + sizeof(WirePose));
    CHECK(packetBytes(3) == 16 + 3 * sizeof(WirePose));
    CHECK(packetBytes(static_cast<uint16_t>(kMaxTrackers)) == sizeof(WirePacket));
}

TEST_CASE("protocol: role serials are unique and stable") {
    std::set<std::string> serials;
    for (int i = 0; i < static_cast<int>(Role::Count); ++i) {
        const std::string s = roleSerial(static_cast<Role>(i));
        CHECK(!s.empty());
        CHECK(s != "MN-UNKNOWN");
        CHECK(s.rfind("MN-", 0) == 0);
        CHECK(serials.insert(s).second); // no duplicates
    }
    CHECK(serials.size() == static_cast<size_t>(Role::Count));
    // Out-of-range roles fall back to the sentinel.
    CHECK(std::string(roleSerial(Role::Count)) == "MN-UNKNOWN");
}

TEST_CASE("protocol: WirePacket round-trips through a byte buffer") {
    WirePacket out;
    out.count = 2;
    out.timestamp = 1234.5678;

    out.poses[0].role = static_cast<uint8_t>(Role::Waist);
    out.poses[0].valid = 1;
    out.poses[0].px = 0.1f;
    out.poses[0].py = 1.2f;
    out.poses[0].pz = -0.3f;
    out.poses[0].qw = 0.5f;
    out.poses[0].qx = 0.5f;
    out.poses[0].qy = -0.5f;
    out.poses[0].qz = 0.5f;
    out.poses[0].vx = 1.0f;
    out.poses[0].vy = -2.0f;
    out.poses[0].vz = 3.0f;
    out.poses[0].wx = 0.25f;
    out.poses[0].wy = -0.5f;
    out.poses[0].wz = 0.75f;

    out.poses[1].role = static_cast<uint8_t>(Role::LeftFoot);
    out.poses[1].valid = 1;
    out.poses[1].px = -1.5f;
    out.poses[1].qw = 1.0f;

    // Serialize exactly what the bridge sends: packetBytes(count) bytes.
    const size_t bytes = packetBytes(out.count);
    std::vector<unsigned char> buf(bytes, 0xAB);
    std::memcpy(buf.data(), &out, bytes);

    // Receiver side: a zeroed receive area, partial datagram copied in, then
    // reinterpreted as a packet (WirePacket is trivially copyable).
    static_assert(std::is_trivially_copyable_v<WirePacket>);
    std::vector<unsigned char> rx(sizeof(WirePacket), 0);
    std::memcpy(rx.data(), buf.data(), bytes);
    WirePacket in;
    std::memcpy(&in, rx.data(), sizeof(WirePacket));

    CHECK(in.magic == kMagic);
    CHECK(in.version == kVersion);
    CHECK(in.count == 2);
    CHECK(in.timestamp == doctest::Approx(1234.5678));

    CHECK(in.poses[0].role == static_cast<uint8_t>(Role::Waist));
    CHECK(in.poses[0].valid == 1);
    CHECK(in.poses[0].px == doctest::Approx(0.1f));
    CHECK(in.poses[0].py == doctest::Approx(1.2f));
    CHECK(in.poses[0].pz == doctest::Approx(-0.3f));
    CHECK(in.poses[0].qw == doctest::Approx(0.5f));
    CHECK(in.poses[0].qx == doctest::Approx(0.5f));
    CHECK(in.poses[0].qy == doctest::Approx(-0.5f));
    CHECK(in.poses[0].qz == doctest::Approx(0.5f));
    CHECK(in.poses[0].vx == doctest::Approx(1.0f));
    CHECK(in.poses[0].vy == doctest::Approx(-2.0f));
    CHECK(in.poses[0].vz == doctest::Approx(3.0f));
    CHECK(in.poses[0].wx == doctest::Approx(0.25f));
    CHECK(in.poses[0].wy == doctest::Approx(-0.5f));
    CHECK(in.poses[0].wz == doctest::Approx(0.75f));

    CHECK(in.poses[1].role == static_cast<uint8_t>(Role::LeftFoot));
    CHECK(in.poses[1].px == doctest::Approx(-1.5f));
    CHECK(in.poses[1].qw == doctest::Approx(1.0f));

    // Entries beyond `count` were not transmitted: still default-zeroed.
    CHECK(in.poses[2].role == 0);
    CHECK(in.poses[2].valid == 0);
    CHECK(in.poses[2].qw == doctest::Approx(0.0f));

    // Header fields land at their documented offsets in the raw buffer.
    uint32_t rawMagic = 0;
    uint16_t rawVersion = 0, rawCount = 0;
    double rawTs = 0.0;
    std::memcpy(&rawMagic, buf.data() + 0, sizeof(rawMagic));
    std::memcpy(&rawVersion, buf.data() + 4, sizeof(rawVersion));
    std::memcpy(&rawCount, buf.data() + 6, sizeof(rawCount));
    std::memcpy(&rawTs, buf.data() + 8, sizeof(rawTs));
    CHECK(rawMagic == kMagic);
    CHECK(rawVersion == kVersion);
    CHECK(rawCount == 2);
    CHECK(rawTs == doctest::Approx(1234.5678));
}

TEST_CASE("protocol: heartbeat is header-only") {
    WirePacket hb;
    hb.count = 0;
    hb.timestamp = 42.0;

    std::vector<unsigned char> buf(packetBytes(0));
    std::memcpy(buf.data(), &hb, buf.size());
    CHECK(buf.size() == 16);

    std::vector<unsigned char> rx(sizeof(WirePacket), 0);
    std::memcpy(rx.data(), buf.data(), buf.size());
    WirePacket in;
    std::memcpy(&in, rx.data(), sizeof(WirePacket));
    CHECK(in.magic == kMagic);
    CHECK(in.version == kVersion);
    CHECK(in.count == 0);
    CHECK(in.timestamp == doctest::Approx(42.0));
}
