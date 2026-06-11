#pragma once
// Multi-sensor skeleton fusion.
//
// Tier 1 (implemented): per-joint confidence-weighted fusion + One-Euro
// filtering + optional bone-length constraint. Per-node, per-joint weight:
//
//   w = stateW(track state) * confidence * depthW(range) * occlusionW(view)
//
//   stateW:     Tracked = 1.0, Inferred = cfg.inferredWeight, NotTracked = 0
//   depthW:     1 / (1 + (d / depthNoiseRefMeters)^2)   -- Kinect depth noise
//               grows ~quadratically with range
//   occlusionW: cfg.occlusionPenalty when the joint is on the far side of the
//               torso plane from the sensor (self-occluded view), else 1.0.
//               Torso facing is estimated from that node's own frame.
//
// Tier 2 (roadmap): EKF/UKF over root pose + joint angles with per-sensor
// measurement covariances. The submit() interface is designed so that swap is
// internal.

#include "mn/filters.hpp"
#include "mn/skeleton.hpp"

#include <array>
#include <memory>
#include <string>

namespace mn {

struct FusionConfig {
    double staleSeconds = 0.15;       // drop node frames older than this
    float inferredWeight = 0.25f;     // stateW for Inferred joints
    float minConfidence = 0.05f;      // ignore samples below this confidence
    float depthNoiseRefMeters = 4.0f; // depthW reference range
    float occlusionPenalty = 0.3f;    // weight multiplier for far-side joints
    OneEuroParams filter;             // applied per fused joint position
    bool boneLengthConstraint = true; // enforce calibrated bone lengths
};

// Calibrated per-user bone lengths, indexed by child joint (length to parent).
// boneLengthToParent[Hips] is unused (root).
struct BodyModel {
    std::array<float, kJointCount> boneLengthToParent{};
    bool valid = false;
};

// Accumulates fused (or single-node world-frame) frames and estimates bone
// lengths as the per-bone median over the fed frames.
class BodyModelEstimator {
public:
    void feed(const SkeletonFrame& worldFrame, float minConfidence = 0.5f);
    size_t frameCount() const;
    BodyModel estimate() const;

private:
    std::array<std::vector<float>, kJointCount> samples_;
    size_t frames_ = 0;
};

class FusionEngine {
public:
    explicit FusionEngine(FusionConfig cfg = {});
    ~FusionEngine();

    // Node management. Extrinsic maps node-local -> world.
    void setNode(const std::string& nodeId, const Pose& extrinsic);
    void removeNode(const std::string& nodeId);

    // Thread-safe; called from capture threads with node-local frames.
    void submit(const std::string& nodeId, const SkeletonFrame& localFrame);

    void setBodyModel(const BodyModel& model);

    // Fuse the freshest data into a world-frame skeleton. Returns false when
    // no node has a fresh frame with a body. Called from the tick thread.
    bool fuse(double now, SkeletonFrame& outWorld);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mn
