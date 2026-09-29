#pragma once
// Projection correction + validity checking for the fused world-frame skeleton.
//
// After fusion projects every sensor's node-local skeleton into the shared
// world frame, two things can still be wrong with the result:
//
//   1. A mounting / handedness error flips the whole body across an axis -- the
//      classic "my trackers are mirrored / facing backwards / upside down"
//      Kinect problem. ProjectionCorrection is the user-driven fix: three
//      independent axis flips plus an optional left/right joint swap, applied
//      to the fused skeleton before it becomes trackers.
//
//   2. A bad calibration or a glitching sensor projects joints to impossible
//      places (NaN, metres off the origin, limbs the wrong length).
//      checkProjection is a cheap per-frame validity probe that flags these so
//      the dashboard can warn instead of silently streaming garbage.
//
// Tracker orientations are DERIVED from joint positions downstream (see
// mapping.hpp), so correcting positions here is sufficient to correct the
// resulting trackers' orientations too.

#include "mn/skeleton.hpp"

#include <string>

namespace mn {

// A user-applied correction to the fused world skeleton. Each flip negates that
// world axis for every joint; swapLR exchanges the left/right joint pairs.
//
// Recipes for the common Kinect failure modes:
//   - Facing backwards (180 deg yaw): flipX + flipZ  (a proper rotation, so
//     handedness is preserved and no left/right swap is needed).
//   - Mirrored left <-> right:        flipX + swapLR.
//   - Upside down:                    flipY.
struct ProjectionCorrection {
    bool flipX = false;  // negate world X (mirror left <-> right in space)
    bool flipY = false;  // negate world Y (mirror up <-> down)
    bool flipZ = false;  // negate world Z (mirror forward <-> back)
    bool swapLR = false; // exchange left/right joint samples

    bool active() const { return flipX || flipY || flipZ || swapLR; }

    bool operator==(const ProjectionCorrection& o) const {
        return flipX == o.flipX && flipY == o.flipY && flipZ == o.flipZ && swapLR == o.swapLR;
    }
    bool operator!=(const ProjectionCorrection& o) const { return !(*this == o); }
};

// Apply `c` to a world-frame skeleton in place. Positions only -- orientations
// are derived from positions downstream. No-op when !c.active().
void applyProjectionCorrection(SkeletonFrame& world, const ProjectionCorrection& c);

// Thresholds for checkProjection. Deliberately generous: the goal is to catch
// gross projection errors (a limb flung metres away, NaN, a bone twice its
// plausible length), not to grade fine calibration.
struct ProjectionCheckParams {
    float maxRadiusMeters = 4.0f;  // a tracked joint farther than this from the
                                   // world origin is treated as implausible
    float minBoneMeters = 0.02f;   // shortest plausible parent->child segment
    float maxBoneMeters = 1.00f;   // longest plausible parent->child segment
    float minHeadAboveHips = -0.15f; // head.y - hips.y below this => inverted
                                     // (small negative tolerates lying down)
};

// Result of a per-frame validity probe on a fused world skeleton.
struct ProjectionCheck {
    bool evaluated = false;     // false when the frame carried no body to check
    bool finite = true;         // no NaN/Inf in any tracked joint position
    bool inBounds = true;       // all tracked joints within maxRadiusMeters
    bool bonesPlausible = true; // all connected bone lengths within range
    bool uprightOk = true;      // head sits above hips (catches an upside-down rig)

    int trackedJoints = 0;
    int nonFiniteJoints = 0;
    int outOfBoundsJoints = 0;
    int implausibleBones = 0;
    float worstRadiusMeters = 0.0f;    // farthest tracked joint from the origin
    float worstBoneErrorMeters = 0.0f; // worst bone's distance outside the range
    float headAboveHipsMeters = 0.0f;  // head.y - hips.y; clearly negative =>
                                       // flipped Y

    bool ok() const {
        return !evaluated || (finite && inBounds && bonesPlausible && uprightOk);
    }

    // One-line human-readable summary: "ok", "no body", or a problem list.
    std::string summary() const;
};

ProjectionCheck checkProjection(const SkeletonFrame& world,
                                const ProjectionCheckParams& params = {});

} // namespace mn
