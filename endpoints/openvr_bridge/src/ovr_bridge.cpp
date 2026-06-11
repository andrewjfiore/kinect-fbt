// OpenVR bridge endpoint: serializes mapped tracker poses into mn::wire UDP
// packets for the SteamVR driver (driver/openvr). See mn/protocol.hpp and
// docs/DESIGN.md ("Wire protocol").

#include "mn_ovrbridge/ovr_bridge.hpp"

#include "mn/config.hpp" // Pose <-> JSON
#include "mn/log.hpp"
#include "mn/net.hpp"
#include "mn/protocol.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace mn::ovrbridge {
namespace {

// The wire enum must mirror mn::TrackerRole exactly: the bridge casts roles
// straight into uint8_t and the driver maps them back to serials.
static_assert(static_cast<int>(wire::Role::Waist) == static_cast<int>(TrackerRole::Waist));
static_assert(static_cast<int>(wire::Role::LeftFoot) == static_cast<int>(TrackerRole::LeftFoot));
static_assert(static_cast<int>(wire::Role::RightFoot) == static_cast<int>(TrackerRole::RightFoot));
static_assert(static_cast<int>(wire::Role::Chest) == static_cast<int>(TrackerRole::Chest));
static_assert(static_cast<int>(wire::Role::LeftKnee) == static_cast<int>(TrackerRole::LeftKnee));
static_assert(static_cast<int>(wire::Role::RightKnee) == static_cast<int>(TrackerRole::RightKnee));
static_assert(static_cast<int>(wire::Role::LeftElbow) == static_cast<int>(TrackerRole::LeftElbow));
static_assert(static_cast<int>(wire::Role::RightElbow) ==
              static_cast<int>(TrackerRole::RightElbow));
static_assert(static_cast<int>(wire::Role::Head) == static_cast<int>(TrackerRole::Head));
static_assert(static_cast<int>(wire::Role::Count) == static_cast<int>(TrackerRole::Count));

class OpenVrBridgeEndpoint final : public IServiceEndpoint {
public:
    OpenVrBridgeEndpoint(std::string host, uint16_t port, Pose worldAnchor)
        : host_(std::move(host)), port_(port), anchor_(worldAnchor) {}

    ~OpenVrBridgeEndpoint() override { stop(); }

    std::string name() const override { return "openvr"; }

    bool start() override {
        if (!UdpSocket::globalInit()) {
            lastError_ = "winsock init failed";
            return false;
        }
        if (!socket_.openSend(host_, port_)) {
            lastError_ = "openvr bridge: " + socket_.lastError();
            return false;
        }
        log::info("openvr bridge: sending to ", host_, ":", port_);
        return true;
    }

    void stop() override { socket_.close(); }

    void push(const std::vector<TrackerPose>& trackers, double timestamp) override {
        if (!socket_.isOpen())
            return;

        wire::WirePacket pkt;
        pkt.timestamp = timestamp;

        uint16_t count = 0;
        for (const auto& t : trackers) {
            if (t.role == TrackerRole::Head)
                continue; // the HMD owns the head
            if (!t.valid)
                continue;
            if (count >= wire::kMaxTrackers)
                break;

            // Marionette world -> SteamVR playspace.
            const Vec3 pos = anchor_.apply(t.pose.pos);
            const Quat rot = anchor_.applyRot(t.pose.rot);
            const Vec3 vel = anchor_.rot * t.velocity;
            const Vec3 angVel = anchor_.rot * t.angularVelocity;

            wire::WirePose& w = pkt.poses[count];
            w.role = static_cast<uint8_t>(t.role);
            w.valid = 1;
            w.px = pos.x();
            w.py = pos.y();
            w.pz = pos.z();
            w.qw = rot.w();
            w.qx = rot.x();
            w.qy = rot.y();
            w.qz = rot.z();
            w.vx = vel.x();
            w.vy = vel.y();
            w.vz = vel.z();
            w.wx = angVel.x();
            w.wy = angVel.y();
            w.wz = angVel.z();
            ++count;
        }

        // count == 0 is the heartbeat: keeps the driver's link alive without
        // claiming any tracker poses.
        pkt.count = count;
        socket_.send(&pkt, wire::packetBytes(count));
    }

    std::string lastError() const override { return lastError_; }

private:
    std::string host_;
    uint16_t port_ = wire::kDefaultPort;
    Pose anchor_; // Marionette world -> SteamVR playspace
    UdpSocket socket_;
    std::string lastError_;
};

std::unique_ptr<IServiceEndpoint> makeOpenVrBridge(const nlohmann::json& params,
                                                   std::string& error) {
    try {
        const std::string host = params.value("host", std::string("127.0.0.1"));
        const int port = params.value("port", static_cast<int>(wire::kDefaultPort));
        if (port <= 0 || port > 65535) {
            error = "openvr endpoint: port out of range: " + std::to_string(port);
            return nullptr;
        }
        Pose anchor; // identity until the playspace is calibrated
        if (auto it = params.find("world_anchor"); it != params.end())
            anchor = it->get<Pose>();
        return std::make_unique<OpenVrBridgeEndpoint>(host, static_cast<uint16_t>(port), anchor);
    } catch (const std::exception& e) {
        error = std::string("openvr endpoint: bad params: ") + e.what();
        return nullptr;
    }
}

} // namespace

void registerEndpoints(EndpointRegistry& reg) {
    reg.add("openvr", &makeOpenVrBridge);
}

} // namespace mn::ovrbridge
