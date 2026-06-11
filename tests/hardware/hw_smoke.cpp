// Hardware smoke test for real Kinect sensors (plain main, no doctest).
//
// For each compiled backend it enumerates and starts every present sensor
// (kinect_v2: the single one the SDK supports; kinect_v1: indexes until
// enumeration runs out), captures ~6 s per sensor, and prints a plain-ASCII
// summary table. When two or more sensors are present they are then run
// SIMULTANEOUSLY for another 6 s - the real multi-sensor rig check - and the
// combined rates are reported.
//
// Exit codes (see tests/hardware/CMakeLists.txt, SKIP_RETURN_CODE 77):
//   0   every present sensor delivered frames
//   1   hard failure: a sensor is present but start() failed, or it started
//       and delivered nothing
//   77  skipped: no backend compiled in, or no sensor responded
//
// Body presence ("body" column) is informational only: an empty room still
// passes, because both backends emit ~1 Hz keepalive frames with no body.

#include "mn/capture.hpp"
#include "mn/clock.hpp"

#ifdef MN_HAS_KINECT_V2
#include "mn_kinect2/kinect2.hpp"
#endif
#ifdef MN_HAS_KINECT_V1
#include "mn_kinect1/kinect1.hpp"
#endif

#include <cstdio>

#if !defined(MN_HAS_KINECT_V2) && !defined(MN_HAS_KINECT_V1)

int main() {
    std::printf("SKIP: no Kinect capture backend compiled into this build (install the Kinect\n"
                "SDK 2.0 and/or 1.8, reconfigure, and rebuild).\n"
                "Then connect a Kinect and re-run: ctest -L hardware\n");
    return 77;
}

#else // at least one backend compiled

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr double kCaptureSeconds = 6.0;
constexpr int kExitSkip = 77;

#ifdef MN_HAS_KINECT_V2
// kinect_v2's start() reports this when no sensor answers (see kinect2.cpp).
constexpr const char* kV2NotPresent = "no Kinect v2 sensor present";
#endif
#ifdef MN_HAS_KINECT_V1
// kinect_v1's start() range check reports this past the last live index (the
// factory itself never fails for a non-negative index, so enumeration is
// terminated by this start() error rather than by node creation).
constexpr const char* kV1NotPresent = "out of range";
constexpr int kMaxV1Index = 8; // enumeration safety cap
#endif

struct SensorSpec {
    std::string label; // table label: "kinect_v2", "kinect_v1[0]", ...
    std::string id;    // node id used with the registry
    std::string type;
    nlohmann::json params;
    const char* notPresentNeedle;
};

struct RunResult {
    bool created = false;
    bool started = false;
    bool notPresent = false; // start() failed with the backend's not-present error
    uint64_t frames = 0;
    bool body = false;
    std::string error;
};

struct Counter {
    std::atomic<uint64_t> frames{0};
    std::atomic<bool> body{false};
};

void sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

mn::FrameCallback countingCallback(Counter* c) {
    return [c](const mn::NodeDescriptor&, const mn::SkeletonFrame& f) {
        c->frames.fetch_add(1, std::memory_order_relaxed);
        if (f.hasBody)
            c->body.store(true, std::memory_order_relaxed);
    };
}

void printHeader() {
    std::printf("%-16s %8s %7s %5s  %s\n", "sensor", "frames", "fps", "body", "errors");
}

void printRow(const std::string& label, uint64_t frames, double seconds, bool body,
              const std::string& error) {
    std::printf("%-16s %8llu %7.1f %5s  %s\n", label.c_str(),
                static_cast<unsigned long long>(frames),
                static_cast<double>(frames) / seconds, body ? "yes" : "no",
                error.empty() ? "-" : error.c_str());
}

// Create + start one sensor and capture for `seconds`. The node is destroyed
// on return: backends are not required to support start() after stop(), so
// every phase gets a fresh node from the registry.
RunResult runSingle(const mn::NodeRegistry& reg, const SensorSpec& spec, double seconds) {
    RunResult r;
    std::string err;
    auto node = reg.create(spec.type, spec.id, spec.params, err);
    if (!node) {
        r.error = "create failed: " + err;
        return r;
    }
    r.created = true;
    Counter c;
    if (!node->start(countingCallback(&c))) {
        r.error = node->lastError();
        r.notPresent = (r.error.find(spec.notPresentNeedle) != std::string::npos);
        return r;
    }
    r.started = true;
    const double tEnd = mn::nowSeconds() + seconds;
    while (mn::nowSeconds() < tEnd)
        sleepMs(50);
    node->stop(); // joins the capture thread; the counter quiesces here
    r.frames = c.frames.load();
    r.body = c.body.load();
    r.error = node->lastError(); // residual capture errors, informational
    return r;
}

// Run every present sensor at the same time for `seconds`. Returns false on a
// hard failure (a start failed, or a sensor delivered nothing).
bool runSimultaneous(const mn::NodeRegistry& reg, const std::vector<SensorSpec>& specs,
                     double seconds) {
    std::vector<std::unique_ptr<mn::ICaptureNode>> nodes;
    std::vector<std::unique_ptr<Counter>> counters;
    for (const auto& spec : specs) {
        std::string err;
        auto node = reg.create(spec.type, spec.id, spec.params, err);
        if (!node) {
            std::printf("FAIL: %s: create failed in simultaneous phase: %s\n", spec.label.c_str(),
                        err.c_str());
            return false;
        }
        nodes.push_back(std::move(node));
        counters.push_back(std::make_unique<Counter>());
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i]->start(countingCallback(counters[i].get()))) {
            std::printf("FAIL: %s: start failed in simultaneous phase: %s\n",
                        specs[i].label.c_str(), nodes[i]->lastError().c_str());
            for (size_t k = 0; k < i; ++k)
                nodes[k]->stop();
            return false;
        }
    }
    const double tEnd = mn::nowSeconds() + seconds;
    while (mn::nowSeconds() < tEnd)
        sleepMs(50);
    for (auto& n : nodes)
        n->stop();

    bool ok = true;
    uint64_t total = 0;
    printHeader();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const uint64_t frames = counters[i]->frames.load();
        total += frames;
        printRow(specs[i].label, frames, seconds, counters[i]->body.load(),
                 nodes[i]->lastError());
        if (frames == 0)
            ok = false;
    }
    std::printf("%-16s %8llu %7.1f\n", "combined", static_cast<unsigned long long>(total),
                static_cast<double>(total) / seconds);
    if (!ok)
        std::printf("FAIL: at least one sensor delivered no frames while running "
                    "simultaneously\n");
    return ok;
}

} // namespace

int main() {
    mn::NodeRegistry reg;
#ifdef MN_HAS_KINECT_V2
    mn::kinect2::registerNodes(reg);
#endif
#ifdef MN_HAS_KINECT_V1
    mn::kinect1::registerNodes(reg);
#endif

    std::printf("Marionette hardware smoke test: %d s capture per phase\n",
                static_cast<int>(kCaptureSeconds));

    // ---- enumerate and run each sensor on its own -------------------------
    std::vector<SensorSpec> present; // sensors that started and ran
    std::vector<std::pair<SensorSpec, RunResult>> results;
    bool hardFailure = false;

#ifdef MN_HAS_KINECT_V2
    {
        SensorSpec spec{"kinect_v2", "hw-v2", "kinect_v2", nlohmann::json::object(),
                        kV2NotPresent};
        std::printf("-- %s: starting (the SDK supports a single v2 sensor)\n",
                    spec.label.c_str());
        const RunResult r = runSingle(reg, spec, kCaptureSeconds);
        if (r.started) {
            present.push_back(spec);
            results.emplace_back(spec, r);
            if (r.frames == 0)
                hardFailure = true;
        } else if (r.notPresent) {
            std::printf("   %s: not present\n", spec.label.c_str());
        } else {
            std::printf("FAIL: %s: %s\n", spec.label.c_str(), r.error.c_str());
            hardFailure = true;
        }
    }
#endif
#ifdef MN_HAS_KINECT_V1
    for (int idx = 0; idx < kMaxV1Index; ++idx) {
        SensorSpec spec{"kinect_v1[" + std::to_string(idx) + "]",
                        "hw-v1-" + std::to_string(idx), "kinect_v1",
                        nlohmann::json{{"index", idx}}, kV1NotPresent};
        std::printf("-- %s: starting\n", spec.label.c_str());
        const RunResult r = runSingle(reg, spec, kCaptureSeconds);
        if (!r.created) {
            std::printf("   %s: %s (enumeration stops)\n", spec.label.c_str(), r.error.c_str());
            break;
        }
        if (r.notPresent) {
            std::printf("   %s: not present\n", spec.label.c_str());
            break; // indexes are contiguous; nothing past the first miss
        }
        if (!r.started) {
            std::printf("FAIL: %s: %s\n", spec.label.c_str(), r.error.c_str());
            hardFailure = true;
            continue; // a later index may still hold a working sensor
        }
        present.push_back(spec);
        results.emplace_back(spec, r);
        if (r.frames == 0)
            hardFailure = true;
    }
#endif

    if (present.empty() && !hardFailure) {
        std::printf("\nSKIP: no Kinect sensor responded on this machine.\n"
                    "Connect a Kinect and re-run: ctest -L hardware\n");
        return kExitSkip;
    }

    // ---- per-sensor summary ------------------------------------------------
    std::printf("\nper-sensor results (%d s each):\n", static_cast<int>(kCaptureSeconds));
    printHeader();
    for (const auto& [spec, r] : results)
        printRow(spec.label, r.frames, kCaptureSeconds, r.body,
                 r.frames == 0 ? "no frames" : r.error);

    // ---- the real rig check: all present sensors at once -------------------
    if (present.size() >= 2) {
        std::printf("\nsimultaneous run: %d sensors for %d s\n",
                    static_cast<int>(present.size()), static_cast<int>(kCaptureSeconds));
        if (!runSimultaneous(reg, present, kCaptureSeconds))
            hardFailure = true;
    } else {
        std::printf("\n(only one sensor present; skipping the simultaneous multi-sensor run)\n");
    }

    if (hardFailure) {
        std::printf("\nRESULT: FAIL\n");
        return 1;
    }
    std::printf("\nRESULT: OK\n");
    return 0;
}

#endif // backend availability
