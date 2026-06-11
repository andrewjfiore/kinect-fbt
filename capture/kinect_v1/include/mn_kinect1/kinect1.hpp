#pragma once
// Kinect v1 (Xbox 360) capture via the Microsoft Kinect SDK 1.8 skeleton
// pipeline (20 joints, mapped to the common schema; Chest is synthesized).
// SDK 1.8 supports MULTIPLE v1 sensors on one PC - select with "index".
//
// Registered node type "kinect_v1". params:
//   { "index": 0,                // which sensor (enumeration order)
//     "extrinsic": {pos,rot} }   // optional default; CalibrationStore wins

#include "mn/capture.hpp"

namespace mn::kinect1 {

void registerNodes(NodeRegistry& reg);

} // namespace mn::kinect1
