// Kinect v1 capture backend (Microsoft Kinect SDK 1.8).
//
// SDK 1.8 skeleton space is right-handed, +Y up, +Z pointing from the sensor
// toward the user, +X to the sensor's left (= the user's right, since the user
// faces the sensor), positions in meters. Joints are labeled by the user's
// anatomical side (SHOULDER_LEFT is the user's own left shoulder), so the
// user's left lands at -X. That is exactly the mn node-local convention (same
// as Kinect v2 camera space) - positions pass through with NO mirroring.
//
// v1 tracks 20 joints; the mn schema's Chest is synthesized as the midpoint of
// Spine and ShoulderCenter (see mapping table below). HAND_LEFT/RIGHT have no
// schema slot and are dropped (Wrist* covers the FBT use case).

#include "mn_kinect1/kinect1.hpp"

#include "mn/clock.hpp"
#include "mn/log.hpp"

#ifdef _WIN32

// Note: no WIN32_LEAN_AND_MEAN here - NuiSensor.h needs the COM `interface`
// macro and IMediaObject bits that the lean windows.h would exclude.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <NuiApi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace mn::kinect1 {
namespace {

std::string hrHex(HRESULT hr) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
    return buf;
}

// Human-readable text for the NUI HRESULTs an operator is likely to hit.
std::string describeHr(HRESULT hr) {
    std::string what;
    if (hr == E_NUI_DEVICE_NOT_CONNECTED || hr == E_NUI_NOTCONNECTED)
        what = "sensor not connected";
    else if (hr == E_NUI_DEVICE_NOT_READY || hr == E_NUI_NOTREADY)
        what = "sensor not ready (still enumerating, or a part is disconnected)";
    else if (hr == E_NUI_NOTPOWERED)
        what = "sensor not powered (check the AC adapter)";
    else if (hr == E_NUI_DEVICE_IN_USE)
        what = "sensor in use by another process";
    else if (hr == E_NUI_SKELETAL_ENGINE_BUSY)
        what = "skeletal engine in use by another process";
    else if (hr == E_NUI_INSUFFICIENTBANDWIDTH)
        what = "insufficient USB bandwidth (move the sensor to its own USB controller)";
    else if (hr == E_NUI_BADINDEX)
        what = "bad sensor index";
    else if (hr == E_NUI_NOTGENUINE)
        what = "device reported as not genuine";
    if (what.empty())
        return "HRESULT " + hrHex(hr);
    return what + " (HRESULT " + hrHex(hr) + ")";
}

struct JointMapEntry {
    NUI_SKELETON_POSITION_INDEX src;
    Joint dst;
};

// 16 direct mappings; Hips/Spine/Neck/Head come from the center column and
// Chest is synthesized afterwards.
constexpr JointMapEntry kJointMap[] = {
    {NUI_SKELETON_POSITION_HIP_CENTER, Joint::Hips},
    {NUI_SKELETON_POSITION_SPINE, Joint::Spine},
    {NUI_SKELETON_POSITION_SHOULDER_CENTER, Joint::Neck},
    {NUI_SKELETON_POSITION_HEAD, Joint::Head},
    {NUI_SKELETON_POSITION_SHOULDER_LEFT, Joint::ShoulderL},
    {NUI_SKELETON_POSITION_ELBOW_LEFT, Joint::ElbowL},
    {NUI_SKELETON_POSITION_WRIST_LEFT, Joint::WristL},
    {NUI_SKELETON_POSITION_SHOULDER_RIGHT, Joint::ShoulderR},
    {NUI_SKELETON_POSITION_ELBOW_RIGHT, Joint::ElbowR},
    {NUI_SKELETON_POSITION_WRIST_RIGHT, Joint::WristR},
    {NUI_SKELETON_POSITION_HIP_LEFT, Joint::HipL},
    {NUI_SKELETON_POSITION_KNEE_LEFT, Joint::KneeL},
    {NUI_SKELETON_POSITION_ANKLE_LEFT, Joint::AnkleL},
    {NUI_SKELETON_POSITION_FOOT_LEFT, Joint::FootL},
    {NUI_SKELETON_POSITION_HIP_RIGHT, Joint::HipR},
    {NUI_SKELETON_POSITION_KNEE_RIGHT, Joint::KneeR},
    {NUI_SKELETON_POSITION_ANKLE_RIGHT, Joint::AnkleR},
    {NUI_SKELETON_POSITION_FOOT_RIGHT, Joint::FootR},
};

JointSample sampleFrom(const NUI_SKELETON_DATA& data, NUI_SKELETON_POSITION_INDEX idx) {
    JointSample s;
    switch (data.eSkeletonPositionTrackingState[idx]) {
    case NUI_SKELETON_POSITION_TRACKED:
        s.state = TrackState::Tracked;
        s.confidence = 1.0f;
        break;
    case NUI_SKELETON_POSITION_INFERRED:
        s.state = TrackState::Inferred;
        s.confidence = 0.4f;
        break;
    case NUI_SKELETON_POSITION_NOT_TRACKED:
    default:
        s.state = TrackState::NotTracked;
        s.confidence = 0.0f;
        return s; // pos stays zero; do not trust SDK data for untracked joints
    }
    const Vector4& p = data.SkeletonPositions[idx]; // meters, w unused
    s.pos = Vec3(p.x, p.y, p.z);
    return s;
}

// Chest = midpoint(Spine, ShoulderCenter): min(confidences), Inferred if
// either source is Inferred, NotTracked if either source is NotTracked.
JointSample synthesizeChest(const JointSample& spine, const JointSample& neck) {
    JointSample s;
    if (spine.state == TrackState::NotTracked || neck.state == TrackState::NotTracked)
        return s; // NotTracked, conf 0, pos zero
    s.pos = 0.5f * (spine.pos + neck.pos);
    s.confidence = std::min(spine.confidence, neck.confidence);
    s.state = (spine.state == TrackState::Inferred || neck.state == TrackState::Inferred)
                  ? TrackState::Inferred
                  : TrackState::Tracked;
    return s;
}

SkeletonFrame convertSkeleton(const NUI_SKELETON_DATA& data) {
    SkeletonFrame f;
    f.timestamp = nowSeconds();
    f.hasBody = true;
    for (const JointMapEntry& m : kJointMap)
        f[m.dst] = sampleFrom(data, m.src);
    f[Joint::Chest] = synthesizeChest(f[Joint::Spine], f[Joint::Neck]);
    return f;
}

// Among SKELETON_TRACKED entries, prefer the one closest to the sensor by
// HipCenter z; ties (and the single-skeleton case) resolve to the first.
int pickSkeleton(const NUI_SKELETON_FRAME& frame) {
    int best = -1;
    float bestZ = 0.0f;
    for (int i = 0; i < NUI_SKELETON_COUNT; ++i) {
        const NUI_SKELETON_DATA& d = frame.SkeletonData[i];
        if (d.eTrackingState != NUI_SKELETON_TRACKED)
            continue;
        const float z = d.SkeletonPositions[NUI_SKELETON_POSITION_HIP_CENTER].z;
        if (best < 0 || z < bestZ) {
            best = i;
            bestZ = z;
        }
    }
    return best;
}

constexpr DWORD kWaitTimeoutMs = 500;
constexpr double kEmptyFramePeriodSec = 1.0; // hasBody=false cadence with no skeleton

class Kinect1Node final : public ICaptureNode {
public:
    Kinect1Node(std::string id, int index)
        : desc_{std::move(id), "kinect_v1"}, index_(index) {}

    ~Kinect1Node() override { stop(); }

    Kinect1Node(const Kinect1Node&) = delete;
    Kinect1Node& operator=(const Kinect1Node&) = delete;

    const NodeDescriptor& descriptor() const override { return desc_; }

    bool start(FrameCallback cb) override {
        if (isRunning())
            return true;
        stop(); // clear any leftovers from a previous failed start

        int count = 0;
        HRESULT hr = NuiGetSensorCount(&count);
        if (FAILED(hr)) {
            return fail("NuiGetSensorCount failed: " + describeHr(hr));
        }
        if (index_ < 0 || index_ >= count) {
            return fail("sensor index " + std::to_string(index_) + " out of range: " +
                        std::to_string(count) + " Kinect v1 sensor(s) connected (valid indices " +
                        "0.." + std::to_string(count > 0 ? count - 1 : 0) + ")");
        }

        hr = NuiCreateSensorByIndex(index_, &sensor_);
        if (FAILED(hr) || sensor_ == nullptr) {
            sensor_ = nullptr;
            return fail("NuiCreateSensorByIndex(" + std::to_string(index_) +
                        ") failed: " + describeHr(hr));
        }

        hr = sensor_->NuiInitialize(NUI_INITIALIZE_FLAG_USES_SKELETON);
        if (FAILED(hr)) {
            teardown(false);
            return fail("NuiInitialize(USES_SKELETON) failed: " + describeHr(hr));
        }

        // Manual-reset event; the runtime resets it inside NuiSkeletonGetNextFrame.
        frameEvent_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (frameEvent_ == nullptr) {
            teardown(true);
            return fail("CreateEvent failed: error " + std::to_string(GetLastError()));
        }

        hr = sensor_->NuiSkeletonTrackingEnable(frameEvent_, 0);
        if (FAILED(hr)) {
            teardown(true);
            return fail("NuiSkeletonTrackingEnable failed: " + describeHr(hr));
        }

        cb_ = std::move(cb);
        stopRequested_.store(false);
        running_.store(true);
        thread_ = std::thread(&Kinect1Node::captureLoop, this);
        log::info("kinect_v1[", desc_.id, "]: started (sensor index ", index_, " of ", count, ")");
        return true;
    }

    void stop() override {
        stopRequested_.store(true);
        if (frameEvent_ != nullptr)
            SetEvent(frameEvent_); // wake the capture thread immediately
        if (thread_.joinable())
            thread_.join();
        teardown(true);
        running_.store(false);
    }

    bool isRunning() const override { return running_.load(); }

    std::string lastError() const override {
        std::lock_guard<std::mutex> lock(errMutex_);
        return lastError_;
    }

private:
    bool fail(const std::string& msg) {
        setError(msg);
        log::error("kinect_v1[", desc_.id, "]: ", msg);
        return false;
    }

    void setError(const std::string& msg) {
        std::lock_guard<std::mutex> lock(errMutex_);
        lastError_ = msg;
    }

    // Release SDK resources. Only called with the capture thread not running.
    void teardown(bool initialized) {
        if (sensor_ != nullptr) {
            if (initialized) {
                sensor_->NuiSkeletonTrackingDisable();
                sensor_->NuiShutdown();
            }
            sensor_->Release();
            sensor_ = nullptr;
        }
        if (frameEvent_ != nullptr) {
            CloseHandle(frameEvent_);
            frameEvent_ = nullptr;
        }
    }

    void emitEmptyIfDue(double& lastEmptyEmit) {
        const double now = nowSeconds();
        if (now - lastEmptyEmit < kEmptyFramePeriodSec)
            return;
        lastEmptyEmit = now;
        SkeletonFrame f;
        f.timestamp = now;
        f.hasBody = false;
        cb_(desc_, f);
    }

    void captureLoop() {
        double lastEmptyEmit = 0.0;
        unsigned consecutiveFailures = 0;
        while (!stopRequested_.load()) {
            const DWORD wait = WaitForSingleObject(frameEvent_, kWaitTimeoutMs);
            if (stopRequested_.load())
                break;

            if (wait == WAIT_TIMEOUT) {
                emitEmptyIfDue(lastEmptyEmit);
                continue;
            }
            if (wait != WAIT_OBJECT_0) {
                setError("WaitForSingleObject failed: error " + std::to_string(GetLastError()));
                log::error("kinect_v1[", desc_.id, "]: wait on frame event failed; capture stops");
                break;
            }

            NUI_SKELETON_FRAME frame = {};
            const HRESULT hr = sensor_->NuiSkeletonGetNextFrame(0, &frame);
            if (FAILED(hr)) {
                // The manual-reset event may stay signaled after a failed pull;
                // reset it so the loop blocks instead of spinning.
                ResetEvent(frameEvent_);
                if (hr != E_NUI_FRAME_NO_DATA) {
                    setError("NuiSkeletonGetNextFrame failed: " + describeHr(hr));
                    if (consecutiveFailures == 0)
                        log::warn("kinect_v1[", desc_.id, "]: NuiSkeletonGetNextFrame failed: ",
                                  describeHr(hr));
                    ++consecutiveFailures;
                }
                emitEmptyIfDue(lastEmptyEmit);
                continue;
            }
            if (consecutiveFailures > 0) {
                log::info("kinect_v1[", desc_.id, "]: frame pull recovered after ",
                          consecutiveFailures, " failure(s)");
                consecutiveFailures = 0;
            }

            const int best = pickSkeleton(frame);
            if (best < 0) {
                emitEmptyIfDue(lastEmptyEmit);
                continue;
            }
            cb_(desc_, convertSkeleton(frame.SkeletonData[best]));
        }
        running_.store(false);
    }

    NodeDescriptor desc_;
    int index_ = 0;
    FrameCallback cb_;
    INuiSensor* sensor_ = nullptr;
    HANDLE frameEvent_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};
    mutable std::mutex errMutex_;
    std::string lastError_;
};

} // namespace

void registerNodes(NodeRegistry& reg) {
    reg.add("kinect_v1",
            [](const std::string& id, const nlohmann::json& params,
               std::string& error) -> std::unique_ptr<ICaptureNode> {
                int index = 0;
                if (const auto it = params.find("index"); it != params.end()) {
                    if (!it->is_number_integer()) {
                        error = "kinect_v1 node \"" + id +
                                "\": param \"index\" must be an integer";
                        return nullptr;
                    }
                    index = it->get<int>();
                    if (index < 0) {
                        error = "kinect_v1 node \"" + id +
                                "\": param \"index\" must be >= 0, got " + std::to_string(index);
                        return nullptr;
                    }
                }
                // Range against the live sensor count is checked in start(),
                // where the SDK is actually touched.
                return std::make_unique<Kinect1Node>(id, index);
            });
}

} // namespace mn::kinect1

#else // !_WIN32

// The parent CMake gates this target to Windows; this branch only exists so
// the translation unit stays well-formed if it is ever compiled elsewhere.
namespace mn::kinect1 {

void registerNodes(NodeRegistry& reg) {
    reg.add("kinect_v1",
            [](const std::string& id, const nlohmann::json&,
               std::string& error) -> std::unique_ptr<ICaptureNode> {
                error = "kinect_v1 node \"" + id +
                        "\": backend requires Windows (Microsoft Kinect SDK 1.8)";
                return nullptr;
            });
}

} // namespace mn::kinect1

#endif // _WIN32
