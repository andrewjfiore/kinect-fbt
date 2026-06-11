#pragma once
// Rigid (SE(3), no scale) alignment solvers:
//  - solveRigid: Kabsch/Umeyama point-set alignment via Eigen SVD.
//  - PairCalibrationSession: aligns two sensors that observe the same body by
//    matching joint positions across near-simultaneous frames.
//  - solveAnchor: anchors the Marionette world frame to an external frame
//    (SteamVR playspace) from corresponding point samples, the
//    OpenVR-SpaceCalibrator move (hold a tracked controller, match it to the
//    wrist joint the sensors see).

#include "mn/skeleton.hpp"

#include <vector>

namespace mn {

struct RigidFit {
    bool ok = false;
    Pose transform;    // maps src-frame points into dst-frame
    float rmse = 0.0f; // meters, post-fit residual
    size_t samples = 0;
};

// Least-squares R,t with ||R*src[i] + t - dst[i]|| minimized.
// Requires >= 3 pairs spanning a non-degenerate (non-collinear) set.
RigidFit solveRigid(const std::vector<Vec3>& src, const std::vector<Vec3>& dst);

// Standalone (not nested) so a {} default argument is valid on GCC: a nested
// aggregate's NSDMIs cannot be used in default args while the enclosing class
// is incomplete.
struct PairCalibrationOptions {
    float minConfidence = 0.5f;      // both views must exceed this per joint
    float maxTimeDeltaSec = 0.05f;   // frame pairing window
    size_t minSamples = 200;         // point pairs needed before solve()
    std::vector<Joint> joints = {Joint::Head,   Joint::Hips,   Joint::WristL,
                                 Joint::WristR, Joint::AnkleL, Joint::AnkleR};
};

class PairCalibrationSession {
public:
    using Options = PairCalibrationOptions;

    explicit PairCalibrationSession(Options opt = {});

    // Feed the latest frame from each node (each in its own local frame).
    // Pairs are accumulated when both frames are fresh and joints pass the
    // confidence gate.
    void addFramePair(const SkeletonFrame& reference, const SkeletonFrame& target);

    size_t sampleCount() const;

    // Solves target-local -> reference-local. With the reference node's
    // extrinsic E_ref, the target's extrinsic is E_ref ∘ fit.transform.
    RigidFit solve() const;

private:
    Options opt_;
    std::vector<Vec3> refPts_, tgtPts_;
};

// worldPts[i] (Marionette world) correspond to externalPts[i] (e.g. SteamVR).
// Returns world -> external.
RigidFit solveAnchor(const std::vector<Vec3>& worldPts, const std::vector<Vec3>& externalPts);

} // namespace mn
