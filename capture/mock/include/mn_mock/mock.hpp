#pragma once
// Mock + replay capture nodes; the dev/test path that needs no hardware.
//
// Registered node types:
//
//   "mock"   - synthetic body. params:
//     {
//       "pattern": "walk_in_place" | "tpose" | "sway",   // default walk_in_place
//       "rate_hz": 30.0,
//       "noise_m": 0.0,            // gaussian position noise sigma
//       "seed": 1,                 // noise RNG seed (deterministic)
//       "confidence": 0.9,         // reported per-joint confidence
//       "view_pose": {pos,rot},    // simulated sensor extrinsic (local->world);
//                                  // emitted frames = inverse(view_pose) applied
//                                  // to the world-frame ground truth
//       "occlusion_sim": false     // mark far-side-of-torso joints Inferred
//     }
//
//   "replay" - JSONL playback. params:
//     { "file": "frames.jsonl", "loop": true, "speed": 1.0 }
//
// JSONL format (one frame per line):
//   {"t": <double>, "has_body": true,
//    "joints": [[x,y,z, qw,qx,qy,qz, confidence, state], ... kJointCount entries]}

#include "mn/capture.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace mn::mock {

void registerNodes(NodeRegistry& reg);

// Deterministic world-frame ground-truth body at time t. Body stands at the
// origin facing -Z, ~1.7 m tall. Used by the mock node and by tests (fusion
// accuracy is asserted against this).
SkeletonFrame groundTruth(double t, const std::string& pattern = "walk_in_place");

class JsonlRecorder {
public:
    ~JsonlRecorder();
    bool open(const std::string& path);
    void write(const SkeletonFrame& frame);
    void close();
    size_t frameCount() const { return count_; }

private:
    std::FILE* f_ = nullptr;
    size_t count_ = 0;
};

bool loadJsonl(const std::string& path, std::vector<SkeletonFrame>& out);

} // namespace mn::mock
