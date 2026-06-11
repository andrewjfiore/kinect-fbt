// Kinect v2 capture backend (Microsoft Kinect for Windows SDK 2.0).
//
// One sensor per PC (hard SDK limit) -> the factory hands out at most one live
// node. Acquisition is event-driven: SubscribeFrameArrived + a 500 ms
// WaitForSingleObject loop on the capture thread. Event waits deliver frames
// at the sensor's native ~30 Hz cadence with no polling jitter and zero CPU
// between frames; the 500 ms timeout bounds stop() latency and doubles as the
// tick for the ~1 Hz empty keepalive frames emitted while nobody is tracked.
//
// --- Coordinate frames and handedness ---------------------------------------
// Kinect v2 camera space (CameraSpacePoint, per the SDK 2.0 coordinate-mapping
// docs): origin at the IR sensor center, units meters, +Y up, +Z along the
// optical axis from the sensor toward the user, +X to the SENSOR'S left
// (which is the user's right when the user faces the sensor).
// Handedness check: with +Y up and +Z toward the user, the right-handed
// completion is X = Y cross Z = "up cross toward-user" = the sensor's left.
// So camera space IS right-handed and matches the mn node-local convention
// (right-handed, +Y up, +Z sensor -> user) verbatim; see docs/DESIGN.md.
// Joint labels are anatomical, not view-mirrored: JointType_WristLeft is the
// user's physical left wrist (the official programming guide pairs objects on
// the right side of its mirror-displayed UI with JointType::HandRight, i.e.
// the user's real right hand). A user facing the sensor faces -Z, and a
// -Z-facing body in a right-handed +Y-up frame has its anatomical left at
// negative X - exactly where the SDK reports the left wrist. Positions are
// therefore passed through UNCHANGED and labels map straight across
// (WristLeft -> WristL, ...). Negating X here would flip the frame to
// left-handed and mirror the skeleton; it must not be done.

#include "mn_kinect2/kinect2.hpp"

#include "mn/clock.hpp"
#include "mn/log.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4471) // MIDL forward enum declarations in Kinect.h
#endif
#include <Kinect.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace mn::kinect2 {
namespace {

// The SDK's struct is also named Joint; alias it so it cannot be confused
// with mn::Joint inside this namespace.
using SdkJoint = ::Joint;

constexpr const char* kTypeName = "kinect_v2";
constexpr size_t kSdkJointCount = static_cast<size_t>(JointType_Count); // 25
constexpr DWORD kWaitTimeoutMs = 500;       // stop-flag poll bound for the event wait
constexpr double kEmptyIntervalSec = 1.0;   // keepalive cadence while no body is tracked
constexpr int kAvailabilityPollMs = 100;    // IsAvailable poll step during start()
constexpr int kAvailabilityPolls = 30;      // -> up to 3 s for KinectService to attach

// The SDK supports exactly one v2 sensor per PC; enforce one live node.
std::atomic<bool> g_instanceClaimed{false};

std::string hrMsg(const char* stage, HRESULT hr) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "kinect_v2: %s failed (HRESULT 0x%08lX)", stage,
                  static_cast<unsigned long>(hr));
    return buf;
}

template <typename T> void safeRelease(T*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

// 25 SDK joints -> 19-joint mn schema. Hands, thumbs and tips are ignored;
// labels carry straight across (anatomical on both sides, see header comment).
struct JointMapEntry {
    ::JointType src;
    Joint dst;
};
constexpr std::array<JointMapEntry, kJointCount> kJointMap = {{
    {JointType_Head, Joint::Head},
    {JointType_Neck, Joint::Neck},
    {JointType_SpineShoulder, Joint::Chest},
    {JointType_SpineMid, Joint::Spine},
    {JointType_SpineBase, Joint::Hips},
    {JointType_ShoulderLeft, Joint::ShoulderL},
    {JointType_ElbowLeft, Joint::ElbowL},
    {JointType_WristLeft, Joint::WristL},
    {JointType_ShoulderRight, Joint::ShoulderR},
    {JointType_ElbowRight, Joint::ElbowR},
    {JointType_WristRight, Joint::WristR},
    {JointType_HipLeft, Joint::HipL},
    {JointType_KneeLeft, Joint::KneeL},
    {JointType_AnkleLeft, Joint::AnkleL},
    {JointType_FootLeft, Joint::FootL},
    {JointType_HipRight, Joint::HipR},
    {JointType_KneeRight, Joint::KneeR},
    {JointType_AnkleRight, Joint::AnkleR},
    {JointType_FootRight, Joint::FootR},
}};

class Kinect2Node final : public ICaptureNode {
public:
    explicit Kinect2Node(std::string id) : descriptor_{std::move(id), kTypeName} {}

    ~Kinect2Node() override {
        stop();
        g_instanceClaimed.store(false);
    }

    Kinect2Node(const Kinect2Node&) = delete;
    Kinect2Node& operator=(const Kinect2Node&) = delete;

    const NodeDescriptor& descriptor() const override { return descriptor_; }

    bool start(FrameCallback cb) override {
        if (running_.load()) {
            setError("kinect_v2: start() called while already running");
            return false;
        }
        if (!cb) {
            setError("kinect_v2: start() called with a null frame callback");
            return false;
        }
        cb_ = std::move(cb);

        // The Kinect v2 runtime exposes COM-style free-threaded interfaces that
        // do not require CoInitialize; they may be created on any thread and
        // used from the capture thread.
        HRESULT hr = ::GetDefaultKinectSensor(&sensor_);
        if (FAILED(hr) || !sensor_)
            return failStart("GetDefaultKinectSensor", FAILED(hr) ? hr : E_POINTER);

        hr = sensor_->Open();
        if (FAILED(hr))
            return failStart("IKinectSensor::Open", hr);

        IBodyFrameSource* source = nullptr;
        hr = sensor_->get_BodyFrameSource(&source);
        if (FAILED(hr) || !source)
            return failStart("IKinectSensor::get_BodyFrameSource", FAILED(hr) ? hr : E_POINTER);

        hr = source->OpenReader(&reader_);
        safeRelease(source); // the reader keeps the source alive internally
        if (FAILED(hr) || !reader_)
            return failStart("IBodyFrameSource::OpenReader", FAILED(hr) ? hr : E_POINTER);

        hr = reader_->SubscribeFrameArrived(&waitable_);
        if (FAILED(hr))
            return failStart("IBodyFrameReader::SubscribeFrameArrived", hr);

        // Open() succeeds even with no sensor attached: KinectService reports
        // hardware asynchronously via IsAvailable, normally within ~2 s. Poll
        // briefly so "no sensor present" becomes a clean start() failure
        // instead of a silent stream of empty frames forever.
        bool available = false;
        for (int i = 0; i < kAvailabilityPolls; ++i) {
            BOOLEAN avail = 0;
            hr = sensor_->get_IsAvailable(&avail);
            if (FAILED(hr))
                return failStart("IKinectSensor::get_IsAvailable", hr);
            if (avail != 0) {
                available = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kAvailabilityPollMs));
        }
        if (!available) {
            teardown();
            setError("kinect_v2: no Kinect v2 sensor present (IsAvailable stayed false for 3 s "
                     "after Open; check USB 3.0 connection, power, and the KinectMonitor service)");
            return false;
        }

        stop_.store(false);
        running_.store(true);
        thread_ = std::thread([this] { captureLoop(); });
        log::info("kinect_v2[", descriptor_.id, "]: sensor open and available, capture started");
        return true;
    }

    void stop() override {
        stop_.store(true);
        if (thread_.joinable())
            thread_.join();
        teardown();
        if (running_.exchange(false))
            log::info("kinect_v2[", descriptor_.id, "]: stopped");
    }

    bool isRunning() const override { return running_.load(); }

    std::string lastError() const override {
        std::lock_guard<std::mutex> lock(errMutex_);
        return lastError_;
    }

private:
    void setError(std::string msg) {
        std::lock_guard<std::mutex> lock(errMutex_);
        lastError_ = std::move(msg);
    }

    bool failStart(const char* stage, HRESULT hr) {
        setError(hrMsg(stage, hr));
        log::error(lastError());
        teardown();
        return false;
    }

    // Release everything acquired in start(), in reverse acquisition order.
    // Only called with the capture thread not running (or never started).
    void teardown() {
        if (reader_) {
            if (waitable_ != 0) {
                const HRESULT hr = reader_->UnsubscribeFrameArrived(waitable_);
                if (FAILED(hr))
                    log::warn(hrMsg("IBodyFrameReader::UnsubscribeFrameArrived", hr));
                waitable_ = 0;
            }
            safeRelease(reader_);
        }
        if (sensor_) {
            const HRESULT hr = sensor_->Close();
            if (FAILED(hr))
                log::warn(hrMsg("IKinectSensor::Close", hr));
            safeRelease(sensor_);
        }
    }

    // ---- capture thread -----------------------------------------------------

    void captureLoop() {
        const HANDLE evt = reinterpret_cast<HANDLE>(waitable_);
        lastEmit_ = 0.0; // forces an immediate keepalive on the first idle tick
        hadBody_ = false;
        sensorWasAvailable_ = true;

        while (!stop_.load(std::memory_order_relaxed)) {
            const DWORD wr = ::WaitForSingleObject(evt, kWaitTimeoutMs);
            if (wr == WAIT_OBJECT_0) {
                onFrameEvent();
            } else if (wr == WAIT_TIMEOUT) {
                // No body frames at all for 500 ms (sensor unplugged or service
                // hiccup). Keep the node alive with 1 Hz empties; fusion treats
                // the gap as staleness and degrades gracefully.
                checkAvailability();
                maybeEmitEmpty();
            } else {
                const HRESULT hr = HRESULT_FROM_WIN32(::GetLastError());
                setError(hrMsg("WaitForSingleObject on body frame event", hr));
                log::error(lastError(), " - kinect_v2[", descriptor_.id, "] capture loop exiting");
                break;
            }
        }

        for (auto*& body : bodies_)
            safeRelease(body);
        running_.store(false);
    }

    void onFrameEvent() {
        IBodyFrameArrivedEventArgs* args = nullptr;
        HRESULT hr = reader_->GetFrameArrivedEventData(waitable_, &args);
        if (FAILED(hr) || !args)
            return; // spurious wake; nothing to consume

        IBodyFrameReference* ref = nullptr;
        hr = args->get_FrameReference(&ref);
        safeRelease(args);
        if (FAILED(hr) || !ref)
            return;

        IBodyFrame* frame = nullptr;
        hr = ref->AcquireFrame(&frame);
        safeRelease(ref);
        if (FAILED(hr) || !frame)
            return; // E_PENDING: the frame expired before we got to it; normal

        // Refresh (or lazily create) the six IBody slots, then release the
        // frame immediately - holding it stalls the sensor pipeline.
        hr = frame->GetAndRefreshBodyData(static_cast<UINT>(bodies_.size()), bodies_.data());
        safeRelease(frame);
        if (FAILED(hr)) {
            log::warn(hrMsg("IBodyFrame::GetAndRefreshBodyData", hr));
            return;
        }

        SdkJoint joints[kSdkJointCount];
        if (pickClosestTrackedBody(joints)) {
            emitBodyFrame(joints);
        } else {
            if (hadBody_) {
                hadBody_ = false;
                lastEmit_ = 0.0; // body lost: signal fusion immediately, then 1 Hz
                log::debug("kinect_v2[", descriptor_.id, "]: body lost");
            }
            maybeEmitEmpty();
        }
    }

    // Among IsTracked bodies, pick the one whose SpineBase is closest to the
    // sensor (camera-space Euclidean distance). Returns false if none.
    bool pickClosestTrackedBody(SdkJoint (&jointsOut)[kSdkJointCount]) {
        float bestDist = std::numeric_limits<float>::max();
        bool found = false;
        SdkJoint tmp[kSdkJointCount];

        for (IBody* body : bodies_) {
            if (!body)
                continue;
            BOOLEAN tracked = 0;
            if (FAILED(body->get_IsTracked(&tracked)) || tracked == 0)
                continue;
            if (FAILED(body->GetJoints(static_cast<UINT>(kSdkJointCount), tmp)))
                continue;

            const SdkJoint& sb = tmp[JointType_SpineBase];
            // A tracked body with an untracked SpineBase is still eligible,
            // but only as a last resort behind any body with a real root.
            float dist = 1.0e9f;
            if (sb.TrackingState != TrackingState_NotTracked) {
                const CameraSpacePoint& p = sb.Position;
                dist = std::sqrt(p.X * p.X + p.Y * p.Y + p.Z * p.Z);
            }
            if (dist < bestDist) {
                bestDist = dist;
                for (size_t i = 0; i < kSdkJointCount; ++i)
                    jointsOut[i] = tmp[i];
                found = true;
            }
        }
        return found;
    }

    void emitBodyFrame(const SdkJoint (&joints)[kSdkJointCount]) {
        SkeletonFrame out;
        out.hasBody = true;
        for (const JointMapEntry& m : kJointMap) {
            const SdkJoint& sj = joints[static_cast<size_t>(m.src)];
            JointSample& s = out[m.dst];
            // Camera space == node-local frame: pass positions through
            // unchanged (see the handedness analysis at the top of the file).
            s.pos = Vec3(sj.Position.X, sj.Position.Y, sj.Position.Z);
            switch (sj.TrackingState) {
            case TrackingState_Tracked:
                s.state = TrackState::Tracked;
                s.confidence = 1.0f;
                break;
            case TrackingState_Inferred:
                s.state = TrackState::Inferred;
                s.confidence = 0.5f;
                break;
            default:
                s.state = TrackState::NotTracked;
                s.confidence = 0.0f;
                break;
            }
            // rot stays identity, hasRot stays false: SDK joint orientations
            // are discarded by design (fusion derives orientations itself).
        }
        out.timestamp = nowSeconds();
        if (!hadBody_) {
            hadBody_ = true;
            log::debug("kinect_v2[", descriptor_.id, "]: body acquired");
        }
        lastEmit_ = out.timestamp;
        cb_(descriptor_, out);
    }

    // Emit an empty (hasBody=false) frame at most once per kEmptyIntervalSec
    // so fusion staleness logic sees a live but body-less node.
    void maybeEmitEmpty() {
        const double now = nowSeconds();
        if (now - lastEmit_ < kEmptyIntervalSec)
            return;
        SkeletonFrame out;
        out.timestamp = now;
        out.hasBody = false;
        lastEmit_ = now;
        cb_(descriptor_, out);
    }

    // Log (once per transition) when the sensor drops off / comes back.
    void checkAvailability() {
        BOOLEAN avail = 0;
        if (FAILED(sensor_->get_IsAvailable(&avail)))
            return;
        const bool now = (avail != 0);
        if (now != sensorWasAvailable_) {
            sensorWasAvailable_ = now;
            if (now)
                log::info("kinect_v2[", descriptor_.id, "]: sensor available again");
            else
                log::warn("kinect_v2[", descriptor_.id, "]: sensor became unavailable");
        }
    }

    NodeDescriptor descriptor_;
    FrameCallback cb_;

    IKinectSensor* sensor_ = nullptr;
    IBodyFrameReader* reader_ = nullptr;
    WAITABLE_HANDLE waitable_ = 0;
    std::array<IBody*, BODY_COUNT> bodies_{}; // owned by the capture thread

    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};

    // Capture-thread-only state.
    double lastEmit_ = 0.0;
    bool hadBody_ = false;
    bool sensorWasAvailable_ = true;

    mutable std::mutex errMutex_;
    std::string lastError_;
};

} // namespace

void registerNodes(NodeRegistry& reg) {
    reg.add(kTypeName, [](const std::string& id, const nlohmann::json& params,
                          std::string& error) -> std::unique_ptr<ICaptureNode> {
        // "extrinsic" in params is consumed by the calibration/pipeline layer,
        // not by the backend; nothing else is configurable here.
        (void)params;
        if (g_instanceClaimed.exchange(true)) {
            error = "kinect_v2: the Kinect SDK 2.0 supports exactly one v2 sensor per PC and a "
                    "kinect_v2 node already exists; remove the duplicate node (id '" +
                    id + "') from the config or host it on another PC (remote nodes, roadmap)";
            return nullptr;
        }
        return std::make_unique<Kinect2Node>(id);
    });
}

} // namespace mn::kinect2
