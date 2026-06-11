// OSC codec tests: golden bytes, round trips, bundles, malformed input.

#include <doctest/doctest.h>

#include "mn_osc/osc.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void appendU32BE(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
}

void appendBytes(std::vector<uint8_t>& out, const std::vector<uint8_t>& b) {
    out.insert(out.end(), b.begin(), b.end());
}

// Wrap elements into an OSC #bundle: "#bundle\0" + 8-byte timetag + per
// element a big-endian int32 size followed by the element bytes.
std::vector<uint8_t> makeBundle(const std::vector<std::vector<uint8_t>>& elements) {
    std::vector<uint8_t> b;
    const char tag[8] = {'#', 'b', 'u', 'n', 'd', 'l', 'e', '\0'};
    b.insert(b.end(), tag, tag + 8);
    appendU32BE(b, 0);
    appendU32BE(b, 1); // timetag "immediately"
    for (const auto& e : elements) {
        appendU32BE(b, static_cast<uint32_t>(e.size()));
        appendBytes(b, e);
    }
    return b;
}

} // namespace

TEST_CASE("osc encodeMessage golden bytes for /test with one float") {
    const auto buf = mn::osc::encodeMessage("/test", {1.0f});
    // "/test" + NUL + 2 pad NULs (8) | ",f" + 2 pad NULs (4) | 1.0f big-endian
    const uint8_t expected[16] = {'/', 't',  'e', 's', 't', 0x00, 0x00, 0x00,
                                  ',', 'f',  0x00, 0x00,
                                  0x3F, 0x80, 0x00, 0x00};
    REQUIRE(buf.size() == sizeof(expected));
    for (size_t i = 0; i < sizeof(expected); ++i) {
        CAPTURE(i);
        CHECK(buf[i] == expected[i]);
    }
}

TEST_CASE("osc encodeMessage padding for three floats") {
    // "/a" -> 4 bytes; ",fff" (4 chars) still needs a NUL -> padded to 8; 12 arg bytes.
    const auto buf = mn::osc::encodeMessage("/a", {1.0f, 2.0f, 3.0f});
    REQUIRE(buf.size() == 24);
    CHECK(buf[0] == '/');
    CHECK(buf[2] == 0x00);
    CHECK(buf[3] == 0x00);
    CHECK(buf[4] == ',');
    CHECK(buf[7] == 'f');
    CHECK(buf[8] == 0x00); // typetag terminator
    CHECK(buf[11] == 0x00);
}

TEST_CASE("osc encode -> parse round trip") {
    const std::vector<float> args = {0.5f, -1.25f, 3.0f};
    const auto buf = mn::osc::encodeMessage("/tracking/trackers/1/position", args);
    const auto msgs = mn::osc::parsePacket(buf.data(), buf.size());
    REQUIRE(msgs.size() == 1);
    CHECK(msgs[0].address == "/tracking/trackers/1/position");
    REQUIRE(msgs[0].floats.size() == 3);
    // Big-endian float round trip is bit-exact.
    CHECK(msgs[0].floats[0] == args[0]);
    CHECK(msgs[0].floats[1] == args[1]);
    CHECK(msgs[0].floats[2] == args[2]);
}

TEST_CASE("osc zero-argument message round trips") {
    const auto buf = mn::osc::encodeMessage("/ping", {});
    const auto msgs = mn::osc::parsePacket(buf.data(), buf.size());
    REQUIRE(msgs.size() == 1);
    CHECK(msgs[0].address == "/ping");
    CHECK(msgs[0].floats.empty());
}

TEST_CASE("osc parse tolerates trailing bytes after a message") {
    auto buf = mn::osc::encodeMessage("/t", {1.0f, 2.0f});
    buf.push_back(0xDE);
    buf.push_back(0xAD);
    buf.push_back(0xBE);
    const auto msgs = mn::osc::parsePacket(buf.data(), buf.size());
    REQUIRE(msgs.size() == 1);
    REQUIRE(msgs[0].floats.size() == 2);
    CHECK(msgs[0].floats[0] == 1.0f);
    CHECK(msgs[0].floats[1] == 2.0f);
}

TEST_CASE("osc parse bundle with mixed-type message skips non-floats") {
    // Hand-built message: address "/mix", typetags ",ifs", args int32 7,
    // float 2.5f, string "abc". Only the float must come back.
    std::vector<uint8_t> mix = {'/', 'm', 'i', 'x', 0x00, 0x00, 0x00, 0x00,
                                ',', 'i', 'f', 's', 0x00, 0x00, 0x00, 0x00};
    appendU32BE(mix, 7u);          // int32 arg
    appendU32BE(mix, 0x40200000u); // 2.5f big-endian
    const uint8_t str[4] = {'a', 'b', 'c', 0x00};
    mix.insert(mix.end(), str, str + 4);

    const auto rot = mn::osc::encodeMessage("/tracking/trackers/head/rotation", {0.0f, 90.0f, 0.0f});
    const auto bundle = makeBundle({rot, mix});

    const auto msgs = mn::osc::parsePacket(bundle.data(), bundle.size());
    REQUIRE(msgs.size() == 2);
    CHECK(msgs[0].address == "/tracking/trackers/head/rotation");
    REQUIRE(msgs[0].floats.size() == 3);
    CHECK(msgs[0].floats[1] == 90.0f);
    CHECK(msgs[1].address == "/mix");
    REQUIRE(msgs[1].floats.size() == 1);
    CHECK(msgs[1].floats[0] == 2.5f);
}

TEST_CASE("osc parse nested bundle recurses") {
    const auto inner = makeBundle({mn::osc::encodeMessage("/inner", {4.0f})});
    const auto outer = makeBundle({inner, mn::osc::encodeMessage("/outer", {5.0f})});
    const auto msgs = mn::osc::parsePacket(outer.data(), outer.size());
    REQUIRE(msgs.size() == 2);
    CHECK(msgs[0].address == "/inner");
    CHECK(msgs[1].address == "/outer");
    REQUIRE(msgs[0].floats.size() == 1);
    CHECK(msgs[0].floats[0] == 4.0f);
    REQUIRE(msgs[1].floats.size() == 1);
    CHECK(msgs[1].floats[0] == 5.0f);
}

TEST_CASE("osc parse returns empty on malformed input") {
    SUBCASE("empty packet") {
        const uint8_t dummy = 0;
        CHECK(mn::osc::parsePacket(&dummy, 0).empty());
        CHECK(mn::osc::parsePacket(nullptr, 4).empty());
    }
    SUBCASE("not a message or bundle") {
        const uint8_t garbage[8] = {'n', 'o', 'p', 'e', 0x00, 0x00, 0x00, 0x00};
        CHECK(mn::osc::parsePacket(garbage, sizeof(garbage)).empty());
    }
    SUBCASE("address missing NUL terminator") {
        const uint8_t noNul[4] = {'/', 'a', 'b', 'c'};
        CHECK(mn::osc::parsePacket(noNul, sizeof(noNul)).empty());
    }
    SUBCASE("truncated float argument") {
        auto buf = mn::osc::encodeMessage("/test", {1.0f});
        buf.resize(buf.size() - 2);
        CHECK(mn::osc::parsePacket(buf.data(), buf.size()).empty());
    }
    SUBCASE("unknown typetag") {
        // ",q" is not an OSC type we can size; the packet is unusable.
        const uint8_t bad[12] = {'/', 'x', 0x00, 0x00, ',', 'q', 0x00, 0x00,
                                 0x00, 0x00, 0x00, 0x00};
        CHECK(mn::osc::parsePacket(bad, sizeof(bad)).empty());
    }
    SUBCASE("bundle element size exceeds packet") {
        std::vector<uint8_t> b;
        const char tag[8] = {'#', 'b', 'u', 'n', 'd', 'l', 'e', '\0'};
        b.insert(b.end(), tag, tag + 8);
        appendU32BE(b, 0);
        appendU32BE(b, 1);    // timetag
        appendU32BE(b, 100u); // claims 100 bytes...
        appendU32BE(b, 0u);   // ...but only 4 follow
        CHECK(mn::osc::parsePacket(b.data(), b.size()).empty());
    }
}
