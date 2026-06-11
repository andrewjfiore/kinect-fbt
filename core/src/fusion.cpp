// Tier-1 multi-sensor fusion: per-joint confidence-weighted averaging with
// depth and self-occlusion weighting, One-Euro smoothing, and an optional
// bone-length constraint pass. See docs/DESIGN.md "Fusion (tier 1)".

#include "mn/fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mn {

namespace {

// Joints farther than this beyond the torso plane (on the side away from the
// sensor) count as self-occluded for that view.
constexpr float kFarSideMarginMeters = 0.10f;

float stateWeight(TrackState s, float inferredWeight) {
    switch (s) {
    case TrackState::Tracked:
        return 1.0f;
    case TrackState::Inferred:
        return inferredWeight;
    default:
        return 0.0f;
    }
}

// Kinect depth noise grows roughly quadratically with range.
float depthWeight(float rangeMeters, float refMeters) {
    if (refMeters <= 0.0f)
        return 1.0f;
    const float r = rangeMeters / refMeters;
    return 1.0f / (1.0f + r * r);
}

// Torso facing normal in a node's local frame, estimated from that node's own
// frame as cross(hipR - hipL, spine - hips) and sign-resolved so it points
// away from the sensor-facing side (the sensor sits at the local origin).
// Returns false when the torso joints do not span a usable plane.
bool torsoPlaneLocal(const SkeletonFrame& local, Vec3& outNormal, Vec3& outOrigin) {
    const JointSample& hips = local[Joint::Hips];
    const JointSample& spine = local[Joint::Spine];
    const JointSample& hipL = local[Joint::HipL];
    const JointSample& hipR = local[Joint::HipR];
    if (hips.state == TrackState::NotTracked || spine.state == TrackState::NotTracked ||
        hipL.state == TrackState::NotTracked || hipR.state == TrackState::NotTracked)
        return false;
    Vec3 n = (hipR.pos - hipL.pos).cross(spine.pos - hips.pos);
    const float len = n.norm();
    if (len < 1e-6f)
        return false;
    n /= len;
    // Direction from the body toward the sensor is -hips.pos; flip the normal
    // so it points to the far side (away from the sensor).
    if (n.dot(-hips.pos) > 0.0f)
        n = -n;
    outNormal = n;
    outOrigin = hips.pos;
    return true;
}

// Joints ordered so every parent precedes its children (Hips first).
const std::array<Joint, kJointCount>& topDownOrder() {
    static const std::array<Joint, kJointCount> order = [] {
        std::array<int, kJointCount> depth{};
        for (size_t i = 0; i < kJointCount; ++i) {
            int d = 0;
            Joint j = static_cast<Joint>(i);
            while (j != Joint::Hips && d < static_cast<int>(kJointCount)) {
                j = jointParent(j);
                ++d;
            }
            depth[i] = d;
        }
        std::array<Joint, kJointCount> o{};
        for (size_t i = 0; i < kJointCount; ++i)
            o[i] = static_cast<Joint>(i);
        std::stable_sort(o.begin(), o.end(), [&](Joint a, Joint b) {
            return depth[static_cast<size_t>(a)] < depth[static_cast<size_t>(b)];
        });
        return o;
    }();
    return order;
}

} // namespace

// ---------------------------------------------------------------------------
// BodyModelEstimator

void BodyModelEstimator::feed(const SkeletonFrame& worldFrame, float minConfidence) {
    if (!worldFrame.hasBody)
        return;
    bool used = false;
    for (size_t i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        if (j == Joint::Hips)
            continue; // root, no bone to a parent
        const JointSample& child = worldFrame[j];
        const JointSample& parent = worldFrame[jointParent(j)];
        if (child.state == TrackState::NotTracked || parent.state == TrackState::NotTracked)
            continue;
        if (child.confidence < minConfidence || parent.confidence < minConfidence)
            continue;
        samples_[i].push_back((child.pos - parent.pos).norm());
        used = true;
    }
    if (used)
        ++frames_;
}

size_t BodyModelEstimator::frameCount() const {
    return frames_;
}

BodyModel BodyModelEstimator::estimate() const {
    BodyModel model;
    bool any = false;
    for (size_t i = 0; i < kJointCount; ++i) {
        if (samples_[i].empty())
            continue;
        std::vector<float> v = samples_[i];
        const auto mid = static_cast<std::ptrdiff_t>(v.size() / 2);
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        model.boneLengthToParent[i] = v[static_cast<size_t>(mid)];
        any = true;
    }
    model.valid = any;
    return model;
}

// ---------------------------------------------------------------------------
// FusionEngine

struct FusionEngine::Impl {
    // Latest-frame mailbox; one per node, written by that node's capture
    // thread and drained by the tick thread.
    struct Mailbox {
        std::mutex mtx;
        Pose extrinsic;       // node-local -> world
        SkeletonFrame frame;  // latest node-local frame
        bool hasFrame = false;
    };

    FusionConfig cfg;

    std::mutex nodesMtx; // guards the map structure
    std::unordered_map<std::string, std::shared_ptr<Mailbox>> nodes;

    std::mutex stateMtx; // guards bodyModel (set from app thread, read on tick)
    BodyModel bodyModel;

    // Tick-thread only.
    std::array<OneEuroVec3, kJointCount> filters;
};

FusionEngine::FusionEngine(FusionConfig cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;
    for (auto& f : impl_->filters)
        f.setParams(cfg.filter);
}

FusionEngine::~FusionEngine() = default;

void FusionEngine::setNode(const std::string& nodeId, const Pose& extrinsic) {
    std::shared_ptr<Impl::Mailbox> box;
    {
        std::lock_guard<std::mutex> lock(impl_->nodesMtx);
        auto& slot = impl_->nodes[nodeId];
        if (!slot)
            slot = std::make_shared<Impl::Mailbox>();
        box = slot;
    }
    std::lock_guard<std::mutex> boxLock(box->mtx);
    box->extrinsic = extrinsic;
}

void FusionEngine::removeNode(const std::string& nodeId) {
    std::lock_guard<std::mutex> lock(impl_->nodesMtx);
    impl_->nodes.erase(nodeId);
}

void FusionEngine::submit(const std::string& nodeId, const SkeletonFrame& localFrame) {
    std::shared_ptr<Impl::Mailbox> box;
    {
        std::lock_guard<std::mutex> lock(impl_->nodesMtx);
        const auto it = impl_->nodes.find(nodeId);
        if (it == impl_->nodes.end())
            return; // unknown node: no extrinsic, the frame cannot be placed in world
        box = it->second;
    }
    std::lock_guard<std::mutex> boxLock(box->mtx);
    box->frame = localFrame;
    box->hasFrame = true;
}

void FusionEngine::setBodyModel(const BodyModel& model) {
    std::lock_guard<std::mutex> lock(impl_->stateMtx);
    impl_->bodyModel = model;
}

bool FusionEngine::fuse(double now, SkeletonFrame& outWorld) {
    const FusionConfig& cfg = impl_->cfg;

    struct View {
        Pose extrinsic;
        SkeletonFrame frame;
        bool hasPlane = false;
        Vec3 planeNormal{Vec3::Zero()};
        Vec3 planeOrigin{Vec3::Zero()};
    };
    std::vector<View> views;
    {
        std::lock_guard<std::mutex> lock(impl_->nodesMtx);
        views.reserve(impl_->nodes.size());
        for (const auto& entry : impl_->nodes) {
            Impl::Mailbox& box = *entry.second;
            std::lock_guard<std::mutex> boxLock(box.mtx);
            if (!box.hasFrame || !box.frame.hasBody)
                continue;
            if (now - box.frame.timestamp > cfg.staleSeconds)
                continue;
            View v;
            v.extrinsic = box.extrinsic;
            v.frame = box.frame;
            views.push_back(std::move(v));
        }
    }
    for (View& v : views)
        v.hasPlane = torsoPlaneLocal(v.frame, v.planeNormal, v.planeOrigin);

    outWorld = SkeletonFrame{};
    outWorld.timestamp = now;

    bool anyJoint = false;
    for (size_t i = 0; i < kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        float wsum = 0.0f;
        Vec3 acc = Vec3::Zero();
        bool anyTracked = false;
        for (const View& v : views) {
            const JointSample& s = v.frame[j];
            if (s.state == TrackState::NotTracked || s.confidence < cfg.minConfidence)
                continue;
            // Range for the depth-noise model is the node-LOCAL distance.
            float w = stateWeight(s.state, cfg.inferredWeight) * s.confidence *
                      depthWeight(s.pos.norm(), cfg.depthNoiseRefMeters);
            if (v.hasPlane && v.planeNormal.dot(s.pos - v.planeOrigin) > kFarSideMarginMeters)
                w *= cfg.occlusionPenalty;
            if (w <= 0.0f)
                continue;
            acc += w * v.extrinsic.apply(s.pos);
            wsum += w;
            if (s.state == TrackState::Tracked)
                anyTracked = true;
        }
        if (wsum <= 0.0f)
            continue;
        JointSample& o = outWorld[j];
        o.pos = acc / wsum;
        o.state = anyTracked ? TrackState::Tracked : TrackState::Inferred;
        o.confidence = std::min(1.0f, wsum);
        o.rot = Quat::Identity(); // sensor orientations are discarded
        o.hasRot = false;
        anyJoint = true;
    }

    if (!anyJoint) {
        // Body lost: restart the temporal filters so a reacquired body does
        // not get dragged toward stale history.
        for (auto& f : impl_->filters)
            f.reset();
        outWorld.hasBody = false;
        return false;
    }
    outWorld.hasBody = true;

    for (size_t i = 0; i < kJointCount; ++i) {
        JointSample& o = outWorld.joints[i];
        if (o.state == TrackState::NotTracked)
            continue;
        o.pos = impl_->filters[i].filter(o.pos, now);
    }

    BodyModel model;
    {
        std::lock_guard<std::mutex> lock(impl_->stateMtx);
        model = impl_->bodyModel;
    }
    if (cfg.boneLengthConstraint && model.valid) {
        // Single top-down pass from Hips: clamp each child onto its calibrated
        // bone length along the fused direction to its (already constrained)
        // parent.
        for (Joint j : topDownOrder()) {
            if (j == Joint::Hips)
                continue;
            const float len = model.boneLengthToParent[static_cast<size_t>(j)];
            JointSample& child = outWorld[j];
            const JointSample& parent = outWorld[jointParent(j)];
            if (len <= 0.0f || child.state == TrackState::NotTracked ||
                parent.state == TrackState::NotTracked)
                continue;
            const Vec3 dir = safeNormalized(child.pos - parent.pos);
            child.pos = parent.pos + dir * len;
        }
    }
    return true;
}

} // namespace mn
