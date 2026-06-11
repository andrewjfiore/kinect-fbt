#pragma once
// One-Euro filter (Casiez, Roussel, Vogel - CHI 2012). The standard
// low-latency jitter/lag tradeoff filter for tracking signals.

#include "mn/math.hpp"

namespace mn {

struct OneEuroParams {
    float minCutoff = 1.0f; // Hz; lower = smoother at rest
    float beta = 0.05f;     // speed coefficient; higher = less lag in motion
    float dCutoff = 1.0f;   // Hz; derivative low-pass cutoff
};

class OneEuroFilter {
public:
    explicit OneEuroFilter(OneEuroParams p = {}) : params_(p) {}
    // `timestamp` in seconds (monotonic). First call returns `value` unchanged.
    float filter(float value, double timestamp);
    void reset();
    void setParams(OneEuroParams p) { params_ = p; }

private:
    OneEuroParams params_;
    bool init_ = false;
    double lastT_ = 0.0;
    float lastX_ = 0.0f;
    float lastDx_ = 0.0f;
};

class OneEuroVec3 {
public:
    explicit OneEuroVec3(OneEuroParams p = {}) : x_(p), y_(p), z_(p) {}
    Vec3 filter(const Vec3& v, double timestamp);
    void reset();
    void setParams(OneEuroParams p);

private:
    OneEuroFilter x_, y_, z_;
};

} // namespace mn
