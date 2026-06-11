#pragma once
// Bridge endpoint feeding the SteamVR driver (driver/openvr) over localhost
// UDP using mn/protocol.hpp packets.
//
// Registered endpoint type "openvr". params:
//   { "host": "127.0.0.1",
//     "port": 24190,                  // mn::wire::kDefaultPort
//     "world_anchor": {pos,rot} }     // Marionette world -> SteamVR playspace;
//                                     // injected by Pipeline from the
//                                     // CalibrationStore (identity default)
//
// TrackerRole::Head entries are NOT forwarded (the HMD owns the head).

#include "mn/endpoint.hpp"

namespace mn::ovrbridge {

void registerEndpoints(EndpointRegistry& reg);

} // namespace mn::ovrbridge
