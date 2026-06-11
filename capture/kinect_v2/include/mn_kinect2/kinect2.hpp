#pragma once
// Kinect v2 (Xbox One) capture via the Microsoft Kinect SDK 2.0 skeleton
// pipeline (25 joints, mapped down to the common schema). The SDK supports
// exactly ONE v2 sensor per PC; additional v2 units require additional PCs
// (remote nodes, roadmap) or the libfreenect2 markerless backend (roadmap).
//
// Registered node type "kinect_v2". params:
//   { "extrinsic": {pos,rot} }   // optional default; CalibrationStore wins

#include "mn/capture.hpp"

namespace mn::kinect2 {

void registerNodes(NodeRegistry& reg);

} // namespace mn::kinect2
