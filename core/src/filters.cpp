// One-Euro filter implementation (Casiez, Roussel, Vogel - CHI 2012).
//
// alpha(dt, cutoff) = 1 / (1 + tau / dt), tau = 1 / (2 * pi * cutoff)
// The derivative of the signal is low-passed at dCutoff, then the adaptive
// cutoff is minCutoff + beta * |dx_filtered|. First sample passes through.

#include "mn/filters.hpp"

#include <cmath>

namespace mn {
namespace {

constexpr float kPi = 3.14159265358979323846f;

// Exponential smoothing factor for one step of length dt at the given cutoff.
// dt must be > 0 (guarded by the caller). A non-positive cutoff degenerates to
// an infinite time constant, i.e. hold the previous value.
float smoothingAlpha(float dt, float cutoff) {
    if (cutoff <= 0.0f) {
        return 0.0f;
    }
    const float tau = 1.0f / (2.0f * kPi * cutoff);
    return 1.0f / (1.0f + tau / dt);
}

} // namespace

float OneEuroFilter::filter(float value, double timestamp) {
    if (!init_) {
        init_ = true;
        lastT_ = timestamp;
        lastX_ = value;
        lastDx_ = 0.0f;
        return value;
    }

    const float dt = static_cast<float>(timestamp - lastT_);
    if (dt <= 0.0f) {
        return lastX_; // non-monotonic or duplicate timestamp: hold
    }
    lastT_ = timestamp;

    // Low-passed derivative of the raw signal.
    const float dx = (value - lastX_) / dt;
    const float alphaD = smoothingAlpha(dt, params_.dCutoff);
    lastDx_ = alphaD * dx + (1.0f - alphaD) * lastDx_;

    // Speed-adaptive cutoff: smooth at rest, responsive in motion.
    const float cutoff = params_.minCutoff + params_.beta * std::abs(lastDx_);
    const float alpha = smoothingAlpha(dt, cutoff);
    lastX_ = alpha * value + (1.0f - alpha) * lastX_;
    return lastX_;
}

void OneEuroFilter::reset() {
    init_ = false;
    lastT_ = 0.0;
    lastX_ = 0.0f;
    lastDx_ = 0.0f;
}

Vec3 OneEuroVec3::filter(const Vec3& v, double timestamp) {
    return Vec3(x_.filter(v.x(), timestamp), y_.filter(v.y(), timestamp),
                z_.filter(v.z(), timestamp));
}

void OneEuroVec3::reset() {
    x_.reset();
    y_.reset();
    z_.reset();
}

void OneEuroVec3::setParams(OneEuroParams p) {
    x_.setParams(p);
    y_.setParams(p);
    z_.setParams(p);
}

} // namespace mn
