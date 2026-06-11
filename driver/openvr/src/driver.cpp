// driver_marionette: SteamVR server driver exposing Marionette's fused body
// trackers as TrackedDeviceClass_GenericTracker devices.
//
// The core app's "openvr" endpoint (endpoints/openvr_bridge) streams
// mn::wire::WirePacket UDP datagrams to 127.0.0.1:<port>. A receiver thread
// stores the latest WirePose per role; RunFrame() (vrserver main loop) lazily
// registers a device the first time a role is seen and then feeds
// TrackedDevicePoseUpdated every frame. Poses older than 0.5 s are marked
// invalid but the devices stay connected, so SteamVR shows them as idle
// rather than dropping them.
//
// API surface verified against openvr v2.5.1 headers + Valve's simplehmd
// sample; role-hint mechanism follows the MIT-licensed SlimeVR-OpenVR-Driver.

#include "openvr_driver.h"

#include "mn/clock.hpp"
#include "mn/net.hpp"
#include "mn/protocol.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {

constexpr const char* kSettingsSection = "driver_marionette";
constexpr double kPoseStaleSeconds = 0.5;
constexpr size_t kRoleCount = static_cast<size_t>(mn::wire::Role::Count);

void driverLog(const std::string& msg) {
    if (vr::VRDriverLog())
        vr::VRDriverLog()->Log(("marionette: " + msg).c_str());
}

// SteamVR tracker role binding written to the "trackers" settings section
// (steamvr.vrsettings); same mechanism real Vive trackers and SlimeVR use.
const char* roleSettingValue(mn::wire::Role r) {
    switch (r) {
    case mn::wire::Role::Waist: return "TrackerRole_Waist";
    case mn::wire::Role::LeftFoot: return "TrackerRole_LeftFoot";
    case mn::wire::Role::RightFoot: return "TrackerRole_RightFoot";
    case mn::wire::Role::Chest: return "TrackerRole_Chest";
    case mn::wire::Role::LeftKnee: return "TrackerRole_LeftKnee";
    case mn::wire::Role::RightKnee: return "TrackerRole_RightKnee";
    case mn::wire::Role::LeftElbow: return "TrackerRole_LeftElbow";
    case mn::wire::Role::RightElbow: return "TrackerRole_RightElbow";
    default: return "";
    }
}

// Prop_ControllerType_String hint; selects the per-role tracker icon/profile.
const char* controllerTypeHint(mn::wire::Role r) {
    switch (r) {
    case mn::wire::Role::Waist: return "vive_tracker_waist";
    case mn::wire::Role::LeftFoot: return "vive_tracker_left_foot";
    case mn::wire::Role::RightFoot: return "vive_tracker_right_foot";
    case mn::wire::Role::Chest: return "vive_tracker_chest";
    case mn::wire::Role::LeftKnee: return "vive_tracker_left_knee";
    case mn::wire::Role::RightKnee: return "vive_tracker_right_knee";
    case mn::wire::Role::LeftElbow: return "vive_tracker_left_elbow";
    case mn::wire::Role::RightElbow: return "vive_tracker_right_elbow";
    default: return "vive_tracker";
    }
}

// Latest wire sample per role, written by the receiver thread.
struct RoleSlot {
    bool seen = false;          // ever received -> device gets registered
    double arrival = 0.0;       // mn::nowSeconds() at receipt
    mn::wire::WirePose pose;    // latest sample
};

// ---------------------------------------------------------------------------
// One virtual generic tracker.
// ---------------------------------------------------------------------------
class MarionetteTracker final : public vr::ITrackedDeviceServerDriver {
public:
    MarionetteTracker(mn::wire::Role role, std::string serial)
        : role_(role), serial_(std::move(serial)) {
        lastPose_ = defaultPose();
    }

    const std::string& serial() const { return serial_; }
    bool isActive() const { return objectId_ != vr::k_unTrackedDeviceIndexInvalid; }

    vr::EVRInitError Activate(uint32_t unObjectId) override {
        objectId_ = unObjectId;
        const vr::PropertyContainerHandle_t props =
            vr::VRProperties()->TrackedDeviceToPropertyContainer(objectId_);

        vr::VRProperties()->SetStringProperty(props, vr::Prop_ModelNumber_String,
                                              "Marionette Tracker");
        vr::VRProperties()->SetStringProperty(props, vr::Prop_SerialNumber_String,
                                              serial_.c_str());
        vr::VRProperties()->SetStringProperty(props, vr::Prop_ManufacturerName_String,
                                              "Marionette");
        vr::VRProperties()->SetStringProperty(props, vr::Prop_RenderModelName_String,
                                              "{htc}/rendermodels/vr_tracker_vive_1_0");
        vr::VRProperties()->SetStringProperty(props, vr::Prop_InputProfilePath_String,
                                              "{htc}/input/vive_tracker_profile.json");
        vr::VRProperties()->SetStringProperty(props, vr::Prop_ControllerType_String,
                                              controllerTypeHint(role_));
        vr::VRProperties()->SetBoolProperty(props, vr::Prop_WillDriftInYaw_Bool, false);
        vr::VRProperties()->SetBoolProperty(props, vr::Prop_DeviceIsWireless_Bool, true);
        vr::VRProperties()->SetBoolProperty(props, vr::Prop_DeviceCanPowerOff_Bool, false);
        vr::VRProperties()->SetBoolProperty(props, vr::Prop_Identifiable_Bool, false);

        // Bind the body role so SteamVR/VRChat pick it up without manual
        // assignment (takes effect on SteamVR restart, like real trackers).
        const char* roleValue = roleSettingValue(role_);
        if (roleValue[0] != '\0') {
            const std::string key = "/devices/marionette/" + serial_;
            vr::VRSettings()->SetString(vr::k_pch_Trackers_Section, key.c_str(), roleValue);
        }

        driverLog("activated " + serial_ + " (id " + std::to_string(unObjectId) + ")");
        return vr::VRInitError_None;
    }

    void Deactivate() override { objectId_ = vr::k_unTrackedDeviceIndexInvalid; }

    void EnterStandby() override {}

    void* GetComponent(const char* /*pchComponentNameAndVersion*/) override { return nullptr; }

    void DebugRequest(const char* /*pchRequest*/, char* pchResponseBuffer,
                      uint32_t unResponseBufferSize) override {
        if (unResponseBufferSize > 0)
            pchResponseBuffer[0] = '\0';
    }

    vr::DriverPose_t GetPose() override {
        std::lock_guard<std::mutex> lock(poseMutex_);
        return lastPose_;
    }

    // Called from RunFrame (vrserver main thread).
    void submitPose(const mn::wire::WirePose& wire, double ageSeconds) {
        if (!isActive())
            return;

        vr::DriverPose_t pose = defaultPose();
        pose.vecPosition[0] = wire.px;
        pose.vecPosition[1] = wire.py;
        pose.vecPosition[2] = wire.pz;
        pose.qRotation.w = wire.qw;
        pose.qRotation.x = wire.qx;
        pose.qRotation.y = wire.qy;
        pose.qRotation.z = wire.qz;
        pose.vecVelocity[0] = wire.vx;
        pose.vecVelocity[1] = wire.vy;
        pose.vecVelocity[2] = wire.vz;
        pose.vecAngularVelocity[0] = wire.wx;
        pose.vecAngularVelocity[1] = wire.wy;
        pose.vecAngularVelocity[2] = wire.wz;

        const bool fresh = (wire.valid != 0) && (ageSeconds <= kPoseStaleSeconds);
        pose.poseIsValid = fresh;
        pose.result =
            fresh ? vr::TrackingResult_Running_OK : vr::TrackingResult_Running_OutOfRange;
        pose.deviceIsConnected = true;

        {
            std::lock_guard<std::mutex> lock(poseMutex_);
            lastPose_ = pose;
        }
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(objectId_, pose,
                                                           sizeof(vr::DriverPose_t));
    }

private:
    static vr::DriverPose_t defaultPose() {
        vr::DriverPose_t p{};
        // Wire poses arrive already in the SteamVR playspace (the bridge
        // applies the world anchor), so both calibration transforms are
        // identity.
        p.qWorldFromDriverRotation.w = 1.0;
        p.qDriverFromHeadRotation.w = 1.0;
        p.qRotation.w = 1.0;
        p.poseTimeOffset = 0.0;
        p.result = vr::TrackingResult_Running_OutOfRange;
        p.poseIsValid = false;
        p.willDriftInYaw = false;
        p.shouldApplyHeadModel = false;
        p.deviceIsConnected = true;
        return p;
    }

    mn::wire::Role role_;
    std::string serial_;
    uint32_t objectId_ = vr::k_unTrackedDeviceIndexInvalid;
    std::mutex poseMutex_;
    vr::DriverPose_t lastPose_{};
};

// ---------------------------------------------------------------------------
// Server provider: UDP receiver thread + lazy device registration.
// ---------------------------------------------------------------------------
class MarionetteProvider final : public vr::IServerTrackedDeviceProvider {
public:
    vr::EVRInitError Init(vr::IVRDriverContext* pDriverContext) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);

        uint16_t port = mn::wire::kDefaultPort;
        vr::EVRSettingsError settingsErr = vr::VRSettingsError_None;
        const int32_t configured = vr::VRSettings()->GetInt32(kSettingsSection, "port",
                                                              &settingsErr);
        if (settingsErr == vr::VRSettingsError_None && configured > 0 && configured <= 65535)
            port = static_cast<uint16_t>(configured);

        if (!mn::UdpSocket::globalInit()) {
            driverLog("socket subsystem init failed");
            return vr::VRInitError_Driver_Failed;
        }
        if (!socket_.openReceive(port, "127.0.0.1")) {
            driverLog("failed to bind udp 127.0.0.1:" + std::to_string(port) + " (" +
                      socket_.lastError() + ")");
            return vr::VRInitError_Driver_Failed;
        }

        stop_.store(false);
        receiver_ = std::thread(&MarionetteProvider::receiveLoop, this);
        driverLog("listening on udp 127.0.0.1:" + std::to_string(port));
        return vr::VRInitError_None;
    }

    void Cleanup() override {
        stop_.store(true);
        if (receiver_.joinable())
            receiver_.join();
        socket_.close();
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }

    const char* const* GetInterfaceVersions() override { return vr::k_InterfaceVersions; }

    void RunFrame() override {
        // Snapshot the receiver state so we never hold the mutex across
        // vrserver host calls.
        std::array<RoleSlot, kRoleCount> slots;
        {
            std::lock_guard<std::mutex> lock(slotsMutex_);
            slots = slots_;
        }
        const double now = mn::nowSeconds();

        for (size_t i = 0; i < kRoleCount; ++i) {
            const auto role = static_cast<mn::wire::Role>(i);
            if (role == mn::wire::Role::Head || !slots[i].seen)
                continue;

            if (!devices_[i]) {
                auto dev = std::make_unique<MarionetteTracker>(role, mn::wire::roleSerial(role));
                if (!vr::VRServerDriverHost()->TrackedDeviceAdded(
                        dev->serial().c_str(), vr::TrackedDeviceClass_GenericTracker,
                        dev.get())) {
                    driverLog("TrackedDeviceAdded failed for " + dev->serial());
                    continue; // retried next frame
                }
                devices_[i] = std::move(dev);
            }
            devices_[i]->submitPose(slots[i].pose, now - slots[i].arrival);
        }
    }

    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    void receiveLoop() {
        mn::wire::WirePacket pkt;
        while (!stop_.load()) {
            const int n = socket_.receive(&pkt, sizeof(pkt), 100);
            if (n == 0)
                continue; // timeout: re-check stop flag
            if (n < 0) {
                // Persistent socket error: avoid a hot spin, keep trying.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (static_cast<size_t>(n) < mn::wire::packetBytes(0))
                continue;
            if (pkt.magic != mn::wire::kMagic || pkt.version != mn::wire::kVersion)
                continue;
            if (pkt.count > mn::wire::kMaxTrackers ||
                static_cast<size_t>(n) < mn::wire::packetBytes(pkt.count))
                continue;

            const double now = mn::nowSeconds();
            std::lock_guard<std::mutex> lock(slotsMutex_);
            for (uint16_t i = 0; i < pkt.count; ++i) {
                const mn::wire::WirePose& wp = pkt.poses[i];
                if (wp.role >= static_cast<uint8_t>(mn::wire::Role::Count) ||
                    wp.role == static_cast<uint8_t>(mn::wire::Role::Head))
                    continue;
                RoleSlot& slot = slots_[wp.role];
                slot.seen = true;
                slot.arrival = now;
                slot.pose = wp;
            }
        }
    }

    mn::UdpSocket socket_;
    std::thread receiver_;
    std::atomic<bool> stop_{false};

    std::mutex slotsMutex_;
    std::array<RoleSlot, kRoleCount> slots_{};

    // Device objects must outlive vrserver's use of them; never freed until
    // the DLL unloads. Index = role.
    std::array<std::unique_ptr<MarionetteTracker>, kRoleCount> devices_{};
};

MarionetteProvider g_provider;

} // namespace

// ---------------------------------------------------------------------------
// Driver entry point.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
#define MN_DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define MN_DLL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

MN_DLL_EXPORT void* HmdDriverFactory(const char* pInterfaceName, int* pReturnCode) {
    if (std::strcmp(vr::IServerTrackedDeviceProvider_Version, pInterfaceName) == 0)
        return &g_provider;

    if (pReturnCode)
        *pReturnCode = vr::VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
