#pragma once
// VRChat OSC Trackers endpoint. Sends tracker poses straight to the headset
// (or PC client) over UDP - the standalone-Quest path, no PCVR stream needed.
//
// Registered endpoint type "osc". params:
//   { "host": "127.0.0.1",     // headset/PC IP
//     "port": 9000,            // VRChat OSC input port
//     "rate_limit_hz": 0 }     // 0 = send on every pipeline tick
//
// Addresses (VRChat OSC Trackers spec):
//   /tracking/trackers/{1..8}/position  (float3)
//   /tracking/trackers/{1..8}/rotation  (float3, Euler degrees, Unity order)
//   /tracking/trackers/head/position + /rotation  (alignment reference)
// Marionette world (right-handed, +Y up, -Z fwd) is converted to Unity's
// left-handed +Z fwd convention before sending. Tracker indices are assigned
// in the order roles appear in the mapping config (Head -> "head").

#include "mn/endpoint.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mn::osc {

void registerEndpoints(EndpointRegistry& reg);

// --- Minimal OSC codec, exposed for tests --------------------------------
// Encode a single OSC message with float32 arguments.
std::vector<uint8_t> encodeMessage(const std::string& address, const std::vector<float>& args);

struct Message {
    std::string address;
    std::vector<float> floats;
};
// Parse a raw OSC packet (single message or #bundle, recursively). Non-float
// arguments are skipped. Returns extracted messages; empty on malformed input.
std::vector<Message> parsePacket(const uint8_t* data, size_t len);

} // namespace mn::osc
