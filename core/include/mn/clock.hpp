#pragma once
#include <chrono>

namespace mn {

// Monotonic seconds. All SkeletonFrame/TrackerPose timestamps use this clock.
inline double nowSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace mn
