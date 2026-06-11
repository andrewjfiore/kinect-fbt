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

    // Stillness gate: mixed-SDK rigs (v1+v2) deliver skeletons with different
    // latencies, so a moving joint pairs frames that are effectively tens of
    // ms apart and the skew becomes position error. A joint contributes only
    // while it moves slower than this in BOTH views (m/s). <= 0 disables.
    float maxJointSpeed = 0.2f;
    // Trimmed re-solve: after the first fit, drop pairs whose residual
    // exceeds trimFactor * median residual and re-solve (two passes).
    // Robust against the occasional garbage joint estimate.
    bool trimOutliers = true;
    float trimFactor = 2.5f;
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
    // When trimming is enabled, RigidFit.samples/rmse describe the inlier set.
    RigidFit solve() const;

private:
    struct MotionTrack {
        Vec3 pos{Vec3::Zero()};
        double t = -1.0;
        bool still = false;
    };

    Options opt_;
    std::vector<Vec3> refPts_, tgtPts_;
    std::array<MotionTrack, kJointCount> refTrack_{};
    std::array<MotionTrack, kJointCount> tgtTrack_{};
};

// worldPts[i] (Marionette world) correspond to externalPts[i] (e.g. SteamVR).
// Returns world -> external.
RigidFit solveAnchor(const std::vector<Vec3>& worldPts, const std::vector<Vec3>& externalPts);

} // namespace mn
