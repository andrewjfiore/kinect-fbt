// VRChat OSC Trackers endpoint (see mn_osc/osc.hpp).
//
// Spec notes (docs.vrchat.com/docs/osc-trackers, verified 2026-06-10):
//   - Addresses: /tracking/trackers/{1..8}/position, /tracking/trackers/{1..8}/rotation,
//     /tracking/trackers/head/position, /tracking/trackers/head/rotation.
//   - Each takes a Vector3 as 3 floats (X, Y, Z), world-space, Unity coordinates
//     (left-handed, +Y up, 1.0f = 1 m). Rotations are Euler angles in DEGREES,
//     applied in Unity order Z, then X, then Y.
//   - VRChat receives OSC on UDP port 9000 by default; individual messages per
//     address (no bundling required by the spec).
//   - VRChat shifts the whole OSC tracking space so head/position aligns with the
//     avatar head bone, so no playspace anchor is needed on this path.

#include "mn_osc/osc.hpp"

#include "mn/log.hpp"
#include "mn/net.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

namespace mn::osc {

// ---------------------------------------------------------------------------
// OSC 1.0 binary codec
// ---------------------------------------------------------------------------

namespace {

constexpr size_t kMaxBundleDepth = 16;

// Padded size of an OSC string of `len` chars (excluding the terminator):
// at least one NUL terminator, then NUL padding to a 4-byte boundary.
size_t paddedStringSize(size_t len) { return (len + 4) & ~static_cast<size_t>(3); }

void appendString(std::vector<uint8_t>& out, const std::string& s) {
    const size_t padded = paddedStringSize(s.size());
    out.insert(out.end(), s.begin(), s.end());
    out.insert(out.end(), padded - s.size(), uint8_t{0});
}

void appendU32BE(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
}

bool readU32BE(const uint8_t* d, size_t len, size_t off, uint32_t& out) {
    if (off + 4 > len)
        return false;
    out = (static_cast<uint32_t>(d[off]) << 24) | (static_cast<uint32_t>(d[off + 1]) << 16) |
          (static_cast<uint32_t>(d[off + 2]) << 8) | static_cast<uint32_t>(d[off + 3]);
    return true;
}

// Read a NUL-terminated, 4-byte-padded OSC string starting at `off`. On
// success stores the string and advances `off` past the padding.
bool readString(const uint8_t* d, size_t len, size_t& off, std::string& out) {
    size_t end = off;
    while (end < len && d[end] != 0)
        ++end;
    if (end >= len)
        return false; // no terminator inside the packet
    const size_t strLen = end - off;
    const size_t consumed = paddedStringSize(strLen);
    if (off + consumed > len)
        return false; // padding bytes missing
    out.assign(reinterpret_cast<const char*>(d + off), strLen);
    off += consumed;
    return true;
}

bool parseInto(const uint8_t* d, size_t len, std::vector<Message>& out, size_t depth);

// Parse one non-bundle message. Trailing bytes after the last argument are
// tolerated (ignored); anything structurally broken fails.
bool parseMessage(const uint8_t* d, size_t len, std::vector<Message>& out) {
    size_t off = 0;
    Message msg;
    if (!readString(d, len, off, msg.address) || msg.address.empty() || msg.address[0] != '/')
        return false;
    if (off == len) {
        // Address only: technically the typetag string is mandatory in OSC 1.0,
        // but legacy senders omit it; treat as a message with no arguments.
        out.push_back(std::move(msg));
        return true;
    }
    std::string tags;
    if (!readString(d, len, off, tags) || tags.empty() || tags[0] != ',')
        return false;
    for (size_t i = 1; i < tags.size(); ++i) {
        switch (tags[i]) {
        case 'f': {
            uint32_t bits = 0;
            if (!readU32BE(d, len, off, bits))
                return false;
            msg.floats.push_back(std::bit_cast<float>(bits));
            off += 4;
            break;
        }
        case 'i': // int32
        case 'r': // RGBA
        case 'c': // ASCII char
        case 'm': // MIDI
            if (off + 4 > len)
                return false;
            off += 4;
            break;
        case 'd': // double
        case 'h': // int64
        case 't': // timetag
            if (off + 8 > len)
                return false;
            off += 8;
            break;
        case 's': // string
        case 'S': { // symbol
            std::string skip;
            if (!readString(d, len, off, skip))
                return false;
            break;
        }
        case 'b': { // blob: int32 size + data padded to 4
            uint32_t blobLen = 0;
            if (!readU32BE(d, len, off, blobLen))
                return false;
            off += 4;
            const size_t padded = (static_cast<size_t>(blobLen) + 3) & ~static_cast<size_t>(3);
            if (off + padded > len)
                return false;
            off += padded;
            break;
        }
        case 'T': // true
        case 'F': // false
        case 'N': // nil
        case 'I': // impulse
            break;    // no argument data
        default:
            return false; // unknown tag: cannot know its size, packet is unusable
        }
    }
    out.push_back(std::move(msg));
    return true;
}

bool parseBundle(const uint8_t* d, size_t len, std::vector<Message>& out, size_t depth) {
    // "#bundle\0" (8) + 64-bit timetag (8) + { int32 size, element }*
    size_t off = 16;
    while (off < len) {
        uint32_t elemLen = 0;
        if (!readU32BE(d, len, off, elemLen))
            return false;
        off += 4;
        if (elemLen == 0 || off + elemLen > len)
            return false;
        if (!parseInto(d + off, elemLen, out, depth + 1))
            return false;
        off += elemLen;
    }
    return true;
}

bool parseInto(const uint8_t* d, size_t len, std::vector<Message>& out, size_t depth) {
    if (depth > kMaxBundleDepth || len < 4)
        return false;
    static const char kBundleTag[8] = {'#', 'b', 'u', 'n', 'd', 'l', 'e', '\0'};
    if (len >= 16 && std::memcmp(d, kBundleTag, sizeof(kBundleTag)) == 0)
        return parseBundle(d, len, out, depth);
    if (d[0] == '/')
        return parseMessage(d, len, out);
    return false;
}

} // namespace

std::vector<uint8_t> encodeMessage(const std::string& address, const std::vector<float>& args) {
    std::vector<uint8_t> out;
    out.reserve(paddedStringSize(address.size()) + paddedStringSize(args.size() + 1) +
                4 * args.size());
    appendString(out, address);
    std::string tags(1, ',');
    tags.append(args.size(), 'f');
    appendString(out, tags);
    for (float f : args)
        appendU32BE(out, std::bit_cast<uint32_t>(f));
    return out;
}

std::vector<Message> parsePacket(const uint8_t* data, size_t len) {
    std::vector<Message> out;
    if (!data || len == 0)
        return {};
    if (!parseInto(data, len, out, 0))
        return {};
    return out;
}

// ---------------------------------------------------------------------------
// Marionette world -> Unity conversion
// ---------------------------------------------------------------------------

namespace {

// Marionette world is right-handed, +Y up, -Z forward (OpenVR). Unity is
// left-handed, +Y up, +Z forward. Both share +X right and +Y up, so the frame
// change is the reflection M = diag(1, 1, -1) (negate Z), with M == M^-1.
//
// Positions: p_unity = M * p_world = (x, y, -z).
Vec3 toUnityPosition(const Vec3& p) { return Vec3(p.x(), p.y(), -p.z()); }

// Rotations: a linear map R expressed in the new basis is R' = M * R * M^-1
// = M R M. A reflection conjugation maps a rotation (axis u, angle a) to
// (axis M*u, angle -a) because reflections reverse rotational orientation.
// For q = (w, x, y, z) = (cos(a/2), sin(a/2) * u):
//   q' = (cos(-a/2), sin(-a/2) * (ux, uy, -uz)) = (w, -x, -y, +z).
// Tracker local axes convert for free: M is applied on the local side too, so
// a Marionette tracker facing world -Z with identity rotation becomes a Unity
// tracker facing world +Z with identity rotation - exactly Unity's convention
// (+Z forward, +Y up).
//
// Euler extraction: VRChat applies Euler angles in Unity order "Z, X, Y",
// i.e. R = Ry(yaw) * Rx(pitch) * Rz(roll) on column vectors. Unity pairs
// left-handed axes with a left-handed (clockwise about +axis) positive
// rotation sense; flipping both leaves the elementary matrices algebraically
// identical to the familiar right-handed forms, so standard algebra applies
// to the components of q'. Multiplying out R = Ry(y) Rx(p) Rz(r):
//   R = [ cy*cr + sy*sp*sr,  -cy*sr + sy*sp*cr,  sy*cp ]
//       [ cp*sr,              cp*cr,             -sp    ]
//       [ -sy*cr + cy*sp*sr,  sy*sr + cy*sp*cr,  cy*cp ]
// so: pitch = asin(-m12), roll = atan2(m10, m11), yaw = atan2(m02, m22),
// with a gimbal fallback (cp ~ 0): roll = 0, yaw = atan2(-m20, m00).
// Returned as (X, Y, Z) = (pitch, yaw, roll) in degrees, the component order
// the /rotation message expects.
Vec3 toUnityEulerDegrees(const Quat& q) {
    Quat qu(q.w(), -q.x(), -q.y(), q.z());
    const float n = qu.norm();
    if (n < 1e-6f)
        qu = Quat::Identity();
    else
        qu.coeffs() /= n;
    const Mat3 m = qu.toRotationMatrix();

    constexpr float kRadToDeg = 57.29577951308232f; // 180 / pi
    const float sp = -m(1, 2);
    float pitch, yaw, roll;
    if (std::abs(sp) > 0.99999f) {
        pitch = std::copysign(1.5707963267948966f, sp);
        roll = 0.0f;
        yaw = std::atan2(-m(2, 0), m(0, 0));
    } else {
        pitch = std::asin(sp);
        roll = std::atan2(m(1, 0), m(1, 1));
        yaw = std::atan2(m(0, 2), m(2, 2));
    }
    return Vec3(pitch * kRadToDeg, yaw * kRadToDeg, roll * kRadToDeg);
}

// ---------------------------------------------------------------------------
// Endpoint
// ---------------------------------------------------------------------------

class OscEndpoint final : public IServiceEndpoint {
public:
    OscEndpoint(std::string host, uint16_t port, double rateLimitHz)
        : host_(std::move(host)), port_(port), rateLimitHz_(rateLimitHz) {}

    std::string name() const override {
        return "osc(" + host_ + ":" + std::to_string(port_) + ")";
    }

    bool start() override {
        if (!sock_.openSend(host_, port_)) {
            lastError_ = "osc: openSend(" + host_ + ":" + std::to_string(port_) +
                         ") failed: " + sock_.lastError();
            return false;
        }
        haveLastSend_ = false;
        return true;
    }

    void stop() override { sock_.close(); }

    void push(const std::vector<TrackerPose>& trackers, double timestamp) override {
        if (!sock_.isOpen())
            return;
        if (rateLimitHz_ > 0.0) {
            if (haveLastSend_ && timestamp - lastSendTime_ < 1.0 / rateLimitHz_)
                return;
            haveLastSend_ = true;
            lastSendTime_ = timestamp;
        }
        for (const TrackerPose& t : trackers) {
            // Assign slots by order of appearance (validity-independent) so the
            // role -> slot mapping is deterministic and stable across pushes.
            const Slot* slot = slotFor(t.role);
            if (!slot || !t.valid)
                continue;
            const Vec3 pos = toUnityPosition(t.pose.pos);
            const Vec3 rot = toUnityEulerDegrees(t.pose.rot);
            if (!pos.allFinite() || !rot.allFinite())
                continue;
            const auto posMsg = encodeMessage(slot->posAddr, {pos.x(), pos.y(), pos.z()});
            sock_.send(posMsg.data(), posMsg.size());
            const auto rotMsg = encodeMessage(slot->rotAddr, {rot.x(), rot.y(), rot.z()});
            sock_.send(rotMsg.data(), rotMsg.size());
        }
    }

    std::string lastError() const override { return lastError_; }

private:
    struct Slot {
        bool assigned = false;
        std::string posAddr;
        std::string rotAddr;
    };

    // Head -> "head"; everything else -> "1".."8" in first-seen order. Returns
    // nullptr when all 8 numeric slots are taken (extra roles are dropped).
    const Slot* slotFor(TrackerRole role) {
        const size_t idx = static_cast<size_t>(role);
        if (idx >= slots_.size())
            return nullptr;
        Slot& s = slots_[idx];
        if (s.assigned)
            return s.posAddr.empty() ? nullptr : &s;
        s.assigned = true;
        if (role == TrackerRole::Head) {
            s.posAddr = "/tracking/trackers/head/position";
            s.rotAddr = "/tracking/trackers/head/rotation";
            return &s;
        }
        if (nextNumericSlot_ > 8) {
            log::warn("osc: more than 8 non-head trackers configured; dropping role ",
                      trackerRoleName(role));
            return nullptr; // s stays assigned with empty addresses = dropped
        }
        const std::string n = std::to_string(nextNumericSlot_++);
        s.posAddr = "/tracking/trackers/" + n + "/position";
        s.rotAddr = "/tracking/trackers/" + n + "/rotation";
        return &s;
    }

    std::string host_;
    uint16_t port_;
    double rateLimitHz_;
    UdpSocket sock_;
    std::string lastError_;
    bool haveLastSend_ = false;
    double lastSendTime_ = 0.0;
    int nextNumericSlot_ = 1;
    std::array<Slot, static_cast<size_t>(TrackerRole::Count)> slots_{};
};

} // namespace

void registerEndpoints(EndpointRegistry& reg) {
    reg.add("osc", [](const nlohmann::json& params,
                      std::string& error) -> std::unique_ptr<IServiceEndpoint> {
        try {
            const std::string host = params.value("host", std::string("127.0.0.1"));
            const int port = params.value("port", 9000);
            const double rate = params.value("rate_limit_hz", 0.0);
            if (host.empty()) {
                error = "osc: host must not be empty";
                return nullptr;
            }
            if (port < 1 || port > 65535) {
                error = "osc: port out of range: " + std::to_string(port);
                return nullptr;
            }
            if (rate < 0.0 || !std::isfinite(rate)) {
                error = "osc: rate_limit_hz must be a finite value >= 0";
                return nullptr;
            }
            return std::make_unique<OscEndpoint>(host, static_cast<uint16_t>(port), rate);
        } catch (const std::exception& e) {
            error = std::string("osc: bad params: ") + e.what();
            return nullptr;
        }
    });
}

} // namespace mn::osc
