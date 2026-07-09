// Marionette command line application.
//
// Subcommands:
//   list-types            print capture node and endpoint types in this build
//   run                   full pipeline: capture -> fusion -> mapping -> endpoints
//   record                run one capture node, dump its frames to JSONL
//   calibrate pair        solve a target sensor's extrinsic against a reference sensor
//   calibrate body        estimate per-user bone lengths from fused frames
//   calibrate playspace   anchor the Marionette world to the SteamVR playspace
//   doctor                probe backends, sensors, config, calibration, ports, SteamVR
//
// Exit codes: 0 success, 1 runtime failure, 2 usage error.

#include "mn/calibration.hpp"
#include "mn/capture.hpp"
#include "mn/clock.hpp"
#include "mn/config.hpp"
#include "mn/endpoint.hpp"
#include "mn/events.hpp"
#include "mn/fusion.hpp"
#include "mn/log.hpp"
#include "mn/net.hpp"
#include "mn/pipeline.hpp"
#include "mn/protocol.hpp"
#include "mn/skeleton.hpp"

#include "mn_mock/mock.hpp"
#include "mn_osc/osc.hpp"
#include "mn_ovrbridge/ovr_bridge.hpp"

#ifdef MN_HAS_KINECT_V2
#include "mn_kinect2/kinect2.hpp"
#endif
#ifdef MN_HAS_KINECT_V1
#include "mn_kinect1/kinect1.hpp"
#endif

#ifdef MN_HAS_DASHBOARD
#include "mn_dashboard/dashboard.hpp"
#endif

#ifdef MN_HAS_OPENVR_CLIENT
#include <openvr.h>
#endif

#if defined(_WIN32)
#include <windows.h>
// shellapi.h must follow windows.h.
#include <shellapi.h>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};

// Open a URL in the user's default browser (best effort, non-blocking). Only
// ever called with our own loopback dashboard URL, so there is nothing to
// escape; a failure just means the user clicks the link we printed instead.
void openInBrowser(const std::string& url) {
#if defined(_WIN32)
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
#if defined(__APPLE__)
    const std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
#else
    const std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
    // Best effort: the return is captured only to satisfy system()'s
    // warn_unused_result; a failure just means the user opens the link we print.
    [[maybe_unused]] const int rc = std::system(cmd.c_str());
#endif
}

} // namespace

extern "C" void mnOnSignal(int) {
    g_stop.store(true);
}

namespace {

using mn::AppConfig;
using mn::CalibrationStore;
using mn::EndpointRegistry;
using mn::FrameCallback;
using mn::ICaptureNode;
using mn::Joint;
using mn::NodeConfigEntry;
using mn::NodeDescriptor;
using mn::NodeRegistry;
using mn::Pose;
using mn::SkeletonFrame;
using mn::Vec3;

// ---------------------------------------------------------------- utilities

void sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string fmtF(double v, int prec) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return std::string(buf);
}

bool parseDouble(const std::string& s, double& out) {
    if (s.empty())
        return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || errno == ERANGE)
        return false;
    out = v;
    return true;
}

bool parseSize(const std::string& s, size_t& out) {
    if (s.empty() || s[0] == '-')
        return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end != s.c_str() + s.size() || errno == ERANGE)
        return false;
    out = static_cast<size_t>(v);
    return true;
}

void printUsage() {
    std::cout
        << "Marionette - multi-Kinect full-body tracking\n"
           "\n"
           "Usage:\n"
           "  marionette list-types\n"
           "  marionette run -c <config.json> [--duration <sec>] [--verbose]\n"
           "                 [--no-dashboard] [--dashboard-port <p>] [--no-open]\n"
           "  marionette record -c <config.json> --node <id> -o <out.jsonl> [--duration <sec>]\n"
           "  marionette calibrate pair -c <config.json> --reference <id> --target <id>\n"
           "                            [--min-samples <n>] [--max-seconds <sec>]\n"
           "  marionette calibrate body -c <config.json> [--seconds <sec>]\n"
           "  marionette calibrate playspace -c <config.json> [--hand left|right]\n"
           "                                 [--seconds <sec>]\n"
           "  marionette doctor [-c <config.json>] [--json]\n"
           "  marionette -h | --help\n"
           "\n"
           "Commands:\n"
           "  list-types           List node and endpoint types compiled into this build.\n"
           "  run                  Run the pipeline: capture -> fusion -> trackers -> endpoints.\n"
           "                       Serves the web dashboard (when built) unless --no-dashboard,\n"
           "                       and opens it in your browser unless --no-open.\n"
           "  record               Run a single capture node, write its frames to a JSONL file.\n"
           "  calibrate pair       Solve a target sensor's extrinsic against a reference sensor.\n"
           "  calibrate body       Estimate per-user bone lengths from fused frames.\n"
           "  calibrate playspace  Anchor the Marionette world to the SteamVR playspace.\n"
           "  doctor               Probe backends, sensors, config, calibration, ports, SteamVR.\n"
           "\n"
           "Exit codes: 0 success, 1 runtime failure, 2 usage error.\n";
}

int usageError(const std::string& msg) {
    mn::log::error(msg);
    printUsage();
    return 2;
}

bool loadConfig(const std::string& path, AppConfig& out) {
    try {
        out = AppConfig::load(path);
        return true;
    } catch (const std::exception& e) {
        mn::log::error("failed to load config '", path, "': ", e.what());
        return false;
    }
}

bool loadStore(const std::string& path, CalibrationStore& out) {
    if (!out.load(path)) {
        mn::log::error("malformed calibration file: ", path);
        return false;
    }
    return true;
}

const NodeConfigEntry* findNode(const AppConfig& cfg, const std::string& id) {
    for (const auto& n : cfg.nodes) {
        if (n.id == id)
            return &n;
    }
    return nullptr;
}

std::string nodeIdList(const AppConfig& cfg) {
    std::string out;
    for (const auto& n : cfg.nodes) {
        if (!out.empty())
            out += ", ";
        out += n.id;
    }
    return out.empty() ? std::string("<none>") : out;
}

// Same resolution order the Pipeline uses: CalibrationStore entry, else node
// params["extrinsic"], else identity (with a warning).
Pose resolveExtrinsic(const CalibrationStore& store, const NodeConfigEntry& entry) {
    if (auto stored = store.nodeExtrinsic(entry.id))
        return *stored;
    if (entry.params.is_object() && entry.params.contains("extrinsic")) {
        try {
            return entry.params.at("extrinsic").get<Pose>();
        } catch (const std::exception& e) {
            mn::log::warn("node '", entry.id, "': bad params.extrinsic (", e.what(),
                          "); using identity");
            return Pose::identity();
        }
    }
    mn::log::warn("node '", entry.id, "': no calibrated extrinsic; using identity");
    return Pose::identity();
}

std::unique_ptr<ICaptureNode> buildNode(const NodeRegistry& reg, const NodeConfigEntry& entry) {
    std::string err;
    auto node = reg.create(entry.type, entry.id, entry.params, err);
    if (!node)
        mn::log::error("failed to create node '", entry.id, "' (type '", entry.type, "'): ", err);
    return node;
}

bool buildAllNodes(const AppConfig& cfg, const NodeRegistry& reg,
                   std::vector<std::unique_ptr<ICaptureNode>>& out) {
    if (cfg.nodes.empty()) {
        mn::log::error("config has no capture nodes");
        return false;
    }
    for (const auto& entry : cfg.nodes) {
        auto node = buildNode(reg, entry);
        if (!node)
            return false;
        out.push_back(std::move(node));
    }
    return true;
}

// Starts every node; on any failure stops the already-started ones and fails.
bool startAllNodes(std::vector<std::unique_ptr<ICaptureNode>>& nodes, const FrameCallback& cb) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i]->start(cb)) {
            mn::log::error("node '", nodes[i]->descriptor().id,
                           "' failed to start: ", nodes[i]->lastError());
            for (size_t k = 0; k < i; ++k)
                nodes[k]->stop();
            return false;
        }
    }
    return true;
}

void stopAllNodes(std::vector<std::unique_ptr<ICaptureNode>>& nodes) {
    for (auto& n : nodes)
        n->stop();
}

// ----------------------------------------------- playspace anchor sampling

#ifdef MN_HAS_OPENVR_CLIENT

// Position of the tracked controller for the left/right hand in the SteamVR
// standing universe, or nullopt while that controller has no valid pose.
std::optional<Vec3> controllerWorldPos(vr::IVRSystem* vrSys, bool left) {
    const vr::ETrackedControllerRole role =
        left ? vr::TrackedControllerRole_LeftHand : vr::TrackedControllerRole_RightHand;
    const vr::TrackedDeviceIndex_t idx = vrSys->GetTrackedDeviceIndexForControllerRole(role);
    if (idx == vr::k_unTrackedDeviceIndexInvalid || idx >= vr::k_unMaxTrackedDeviceCount)
        return std::nullopt;
    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    vrSys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses,
                                           vr::k_unMaxTrackedDeviceCount);
    const vr::TrackedDevicePose_t& devPose = poses[idx];
    if (!devPose.bPoseIsValid)
        return std::nullopt;
    const auto& m = devPose.mDeviceToAbsoluteTracking.m;
    return Vec3(m[0][3], m[1][3], m[2][3]);
}

// Shared playspace-anchor sampling loop, used by both `calibrate playspace`
// and the dashboard wizard. Ticks at ~30 Hz: `pump` runs every tick (the
// offline CLI path fuses its own engine there; pass {} when a running
// pipeline already fuses), a (fused wrist, controller) pair is sampled on
// every 3rd tick (~10 Hz), and near-duplicate controller positions (< 1 cm
// apart) are skipped so a resting controller cannot dominate the fit. The
// collected pairs are then solved into the world -> SteamVR anchor.
// `fusedWrist` returns the current fused wrist position for the chosen hand,
// nullopt while it is untracked. Honors g_stop in addition to `cancel`.
// Returns ok=false with `error` set on failure.
mn::RigidFit collectPlayspaceAnchor(
    vr::IVRSystem* vrSys, const std::string& hand, double seconds,
    const std::function<std::optional<Vec3>()>& fusedWrist, const std::function<void()>& pump,
    const std::function<void(size_t, size_t, const std::string&)>& progress,
    const std::atomic<bool>& cancel, std::string& error) {
    const bool left = (hand == "left");
    std::vector<Vec3> worldPts, steamvrPts;
    bool warnedNoController = false;
    // Expected pair count at the ~10 Hz sampling cadence; progress denominator.
    const size_t expected = static_cast<size_t>(seconds * 10.0) + 1;
    const double tEnd = mn::nowSeconds() + seconds;
    double lastStatus = mn::nowSeconds();
    int tick = 0;
    while (!g_stop.load() && !cancel.load() && mn::nowSeconds() < tEnd) {
        sleepMs(33); // ~30 Hz; sample pairs on every 3rd tick (~10 Hz)
        if (pump)
            pump();
        const double now = mn::nowSeconds();
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            mn::log::info("anchor samples: ", worldPts.size());
            if (progress)
                progress(worldPts.size(), expected, "collecting");
        }
        ++tick;
        if (tick % 3 != 0)
            continue;
        const std::optional<Vec3> wrist = fusedWrist();
        if (!wrist)
            continue;
        const std::optional<Vec3> vrPos = controllerWorldPos(vrSys, left);
        if (!vrPos) {
            if (!warnedNoController) {
                warnedNoController = true;
                mn::log::warn("no ", hand, " controller is tracked yet; waiting for it");
            }
            continue;
        }
        // Skip near-duplicate points so a resting controller cannot dominate the fit.
        if (!steamvrPts.empty() && (*vrPos - steamvrPts.back()).norm() < 0.01f)
            continue;
        worldPts.push_back(*wrist);
        steamvrPts.push_back(*vrPos);
    }
    if (progress)
        progress(worldPts.size(), expected, "solving");
    const mn::RigidFit fit = mn::solveAnchor(worldPts, steamvrPts);
    if (!fit.ok)
        error = "playspace calibration failed (" + std::to_string(worldPts.size()) +
                " samples); wave the controller through a larger, non-flat volume while the "
                "sensors track you";
    return fit;
}

#endif // MN_HAS_OPENVR_CLIENT

// -------------------------------------------------------------- list-types

int cmdListTypes(const NodeRegistry& nodes, const EndpointRegistry& endpoints) {
    std::cout << "capture node types:\n";
    for (const auto& t : nodes.types())
        std::cout << "  " << t << "\n";
    std::cout << "endpoint types:\n";
    for (const auto& t : endpoints.types())
        std::cout << "  " << t << "\n";
    return 0;
}

// --------------------------------------------------------------------- run

int cmdRun(int argc, char** argv, const NodeRegistry& nodeReg, const EndpointRegistry& epReg) {
    std::string configPath;
    double duration = 0.0;    // 0 = run until interrupted
    bool verbose = false;
    bool noDashboard = false;
    bool noOpen = false;      // suppress auto-opening the dashboard in a browser
    size_t dashboardPort = 0; // 0 = keep the configured port
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("run: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--duration") {
            if (++i >= argc || !parseDouble(argv[i], duration) || duration <= 0.0)
                return usageError("run: --duration needs a positive number of seconds");
        } else if (a == "--verbose") {
            verbose = true;
        } else if (a == "--no-dashboard") {
            noDashboard = true;
        } else if (a == "--no-open") {
            noOpen = true;
        } else if (a == "--dashboard-port") {
            if (++i >= argc || !parseSize(argv[i], dashboardPort) || dashboardPort == 0 ||
                dashboardPort > 65535)
                return usageError("run: --dashboard-port needs a port number (1-65535)");
        } else {
            return usageError("run: unknown option '" + a + "'");
        }
    }
    if (configPath.empty())
        return usageError("run: -c <config.json> is required");
    if (verbose)
        mn::log::setLevel(mn::log::Level::Debug);

    AppConfig cfg;
    if (!loadConfig(configPath, cfg))
        return 1;
    if (noDashboard)
        cfg.dashboard.enable = false;
    if (dashboardPort != 0)
        cfg.dashboard.port = static_cast<uint16_t>(dashboardPort);
    CalibrationStore store;
    if (!loadStore(cfg.calibrationFile, store))
        return 1;

    std::string err;
    auto pipe = mn::Pipeline::build(cfg, nodeReg, epReg, store, err);
    if (!pipe) {
        mn::log::error("pipeline build failed: ", err);
        return 1;
    }
    if (!pipe->start()) {
        mn::log::error("pipeline failed to start");
        return 1;
    }
    mn::log::info("pipeline running: ", cfg.nodes.size(), " node(s), ", cfg.endpoints.size(),
                  " endpoint(s), tick ", fmtF(cfg.tickHz, 0), " Hz; Ctrl+C to stop");

#ifdef MN_HAS_DASHBOARD
    std::unique_ptr<mn::dash::DashboardServer> dash;
    if (cfg.dashboard.enable) {
        mn::dash::Options dopt;
        dopt.bind = cfg.dashboard.bind;
        dopt.port = cfg.dashboard.port;
#ifdef MN_HAS_KINECT_V2
        dopt.capabilities["kinect_v2"] = true;
#else
        dopt.capabilities["kinect_v2"] = false;
#endif
#ifdef MN_HAS_KINECT_V1
        dopt.capabilities["kinect_v1"] = true;
#else
        dopt.capabilities["kinect_v1"] = false;
#endif
#ifdef MN_HAS_OPENVR_CLIENT
        dopt.capabilities["openvr_client"] = true;
#else
        dopt.capabilities["openvr_client"] = false;
#endif
        dopt.capabilities["dashboard"] = true;
#ifdef MN_HAS_OPENVR_CLIENT
        // Playspace wizard against the LIVE pipeline: latestFused() supplies
        // the wrist samples and the shared sampling/solve path does the rest.
        dopt.playspace = [pl = pipe.get()](const std::string& hand, double seconds,
                                           mn::dash::ProgressFn progress,
                                           std::atomic<bool>& cancel,
                                           std::string& error) -> mn::RigidFit {
            vr::EVRInitError initErr = vr::VRInitError_None;
            vr::IVRSystem* vrSys = vr::VR_Init(&initErr, vr::VRApplication_Background);
            if (vrSys == nullptr || initErr != vr::VRInitError_None) {
                error = std::string("OpenVR init failed: ") +
                        vr::VR_GetVRInitErrorAsSymbol(initErr) + " (" +
                        vr::VR_GetVRInitErrorAsEnglishDescription(initErr) +
                        "); is SteamVR running?";
                return {};
            }
            struct VrShutdown {
                ~VrShutdown() { vr::VR_Shutdown(); }
            } vrShutdown;
            const Joint wrist = (hand == "left") ? Joint::WristL : Joint::WristR;
            const auto fusedWrist = [pl, wrist]() -> std::optional<Vec3> {
                const SkeletonFrame f = pl->latestFused();
                if (!f.hasBody || f[wrist].state == mn::TrackState::NotTracked)
                    return std::nullopt;
                if (mn::nowSeconds() - f.timestamp > 0.5)
                    return std::nullopt; // stale fuse: the pipeline lost the body
                return f[wrist].pos;
            };
            return collectPlayspaceAnchor(vrSys, hand, seconds, fusedWrist, {},
                                          std::move(progress), cancel, error);
        };
#endif // MN_HAS_OPENVR_CLIENT
        dash = std::make_unique<mn::dash::DashboardServer>(*pipe, store, cfg.calibrationFile,
                                                           std::move(dopt));
        if (dash->start()) {
            mn::log::info("dashboard: ", dash->url());
            // No-CLI launch: bring the dashboard up in the default browser so a
            // double-clicked launcher lands the user straight on the UI. Timed
            // runs (--duration) are scripted/tests, so skip the popup there.
            if (!noOpen && duration <= 0.0) {
                mn::log::info("opening ", dash->url(), " (use --no-open to disable)");
                openInBrowser(dash->url());
            }
        } else {
            mn::log::warn("dashboard failed to start (", dash->lastError(),
                          "); continuing without it");
            dash.reset();
        }
    }
#endif // MN_HAS_DASHBOARD

    const double tStart = mn::nowSeconds();
    double lastStatus = tStart;
    int rc = 0;
    while (!g_stop.load()) {
        if (duration > 0.0 && mn::nowSeconds() - tStart >= duration)
            break;
        sleepMs(50);
        if (!pipe->isRunning()) {
            mn::log::error("pipeline stopped unexpectedly");
            rc = 1;
            break;
        }
        const double now = mn::nowSeconds();
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            const auto s = pipe->stats();
            const double pct =
                (s.ticks > 0)
                    ? 100.0 * static_cast<double>(s.fusedTicks) / static_cast<double>(s.ticks)
                    : 0.0;
            mn::log::info("frames in ", s.framesIn, " | fused ticks ", fmtF(pct, 1),
                          "% | trackers out ", s.trackersOut);
        }
    }
    if (g_stop.load())
        mn::log::info("interrupted, stopping");
#ifdef MN_HAS_DASHBOARD
    if (dash)
        dash->stop();
#endif
    pipe->stop();
    const auto s = pipe->stats();
    const double pct =
        (s.ticks > 0) ? 100.0 * static_cast<double>(s.fusedTicks) / static_cast<double>(s.ticks)
                      : 0.0;
    mn::log::info("done: frames in ", s.framesIn, " | ticks ", s.ticks, " | fused ", fmtF(pct, 1),
                  "% | trackers out ", s.trackersOut);
    return rc;
}

// ------------------------------------------------------------------ record

int cmdRecord(int argc, char** argv, const NodeRegistry& nodeReg) {
    std::string configPath, nodeId, outPath;
    double duration = 0.0; // 0 = record until interrupted
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("record: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--node") {
            if (++i >= argc)
                return usageError("record: --node needs a value");
            nodeId = argv[i];
        } else if (a == "-o" || a == "--out") {
            if (++i >= argc)
                return usageError("record: " + a + " needs a value");
            outPath = argv[i];
        } else if (a == "--duration") {
            if (++i >= argc || !parseDouble(argv[i], duration) || duration <= 0.0)
                return usageError("record: --duration needs a positive number of seconds");
        } else {
            return usageError("record: unknown option '" + a + "'");
        }
    }
    if (configPath.empty() || nodeId.empty() || outPath.empty())
        return usageError("record: -c <config.json>, --node <id> and -o <out.jsonl> are required");

    AppConfig cfg;
    if (!loadConfig(configPath, cfg))
        return 1;
    const NodeConfigEntry* entry = findNode(cfg, nodeId);
    if (!entry) {
        mn::log::error("node '", nodeId, "' is not in the config; available: ", nodeIdList(cfg));
        return 1;
    }
    auto node = buildNode(nodeReg, *entry);
    if (!node)
        return 1;

    mn::mock::JsonlRecorder rec;
    if (!rec.open(outPath)) {
        mn::log::error("cannot open output file: ", outPath);
        return 1;
    }

    std::mutex mtx;
    FrameCallback cb = [&rec, &mtx](const NodeDescriptor&, const SkeletonFrame& f) {
        std::lock_guard<std::mutex> lk(mtx);
        rec.write(f);
    };
    if (!node->start(cb)) {
        mn::log::error("node '", nodeId, "' failed to start: ", node->lastError());
        return 1;
    }
    mn::log::info("recording node '", nodeId, "' to ", outPath, "; Ctrl+C to stop");

    const double tStart = mn::nowSeconds();
    double lastStatus = tStart;
    while (!g_stop.load()) {
        if (duration > 0.0 && mn::nowSeconds() - tStart >= duration)
            break;
        sleepMs(50);
        const double now = mn::nowSeconds();
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            size_t n = 0;
            {
                std::lock_guard<std::mutex> lk(mtx);
                n = rec.frameCount();
            }
            mn::log::info("recorded ", n, " frames");
        }
    }
    node->stop(); // joins the capture thread; no more callbacks after this
    rec.close();
    if (rec.frameCount() == 0)
        mn::log::warn("no frames were captured");
    mn::log::info("wrote ", rec.frameCount(), " frames to ", outPath);
    return 0;
}

// ---------------------------------------------------------- calibrate pair

int cmdCalibratePair(int argc, char** argv, const NodeRegistry& nodeReg) {
    std::string configPath, refId, tgtId;
    mn::PairCalibrationSession::Options opt;
    double maxSeconds = 60.0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("calibrate pair: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--reference") {
            if (++i >= argc)
                return usageError("calibrate pair: --reference needs a value");
            refId = argv[i];
        } else if (a == "--target") {
            if (++i >= argc)
                return usageError("calibrate pair: --target needs a value");
            tgtId = argv[i];
        } else if (a == "--min-samples") {
            size_t n = 0;
            if (++i >= argc || !parseSize(argv[i], n) || n == 0)
                return usageError("calibrate pair: --min-samples needs a positive integer");
            opt.minSamples = n;
        } else if (a == "--max-seconds") {
            if (++i >= argc || !parseDouble(argv[i], maxSeconds) || maxSeconds <= 0.0)
                return usageError("calibrate pair: --max-seconds needs a positive number");
        } else {
            return usageError("calibrate pair: unknown option '" + a + "'");
        }
    }
    if (configPath.empty() || refId.empty() || tgtId.empty())
        return usageError(
            "calibrate pair: -c <config.json>, --reference <id> and --target <id> are required");
    if (refId == tgtId)
        return usageError("calibrate pair: --reference and --target must be different nodes");

    AppConfig cfg;
    if (!loadConfig(configPath, cfg))
        return 1;
    CalibrationStore store;
    if (!loadStore(cfg.calibrationFile, store))
        return 1;

    const NodeConfigEntry* refEntry = findNode(cfg, refId);
    const NodeConfigEntry* tgtEntry = findNode(cfg, tgtId);
    if (!refEntry || !tgtEntry) {
        mn::log::error("node '", (!refEntry ? refId : tgtId),
                       "' is not in the config; available: ", nodeIdList(cfg));
        return 1;
    }
    auto refNode = buildNode(nodeReg, *refEntry);
    if (!refNode)
        return 1;
    auto tgtNode = buildNode(nodeReg, *tgtEntry);
    if (!tgtNode)
        return 1;

    std::mutex mtx;
    std::optional<SkeletonFrame> latestRef;
    mn::PairCalibrationSession session(opt);

    FrameCallback refCb = [&mtx, &latestRef](const NodeDescriptor&, const SkeletonFrame& f) {
        std::lock_guard<std::mutex> lk(mtx);
        latestRef = f;
    };
    FrameCallback tgtCb = [&mtx, &latestRef, &session](const NodeDescriptor&,
                                                       const SkeletonFrame& f) {
        std::lock_guard<std::mutex> lk(mtx);
        if (latestRef)
            session.addFramePair(*latestRef, f);
    };

    if (!refNode->start(refCb)) {
        mn::log::error("node '", refId, "' failed to start: ", refNode->lastError());
        return 1;
    }
    if (!tgtNode->start(tgtCb)) {
        mn::log::error("node '", tgtId, "' failed to start: ", tgtNode->lastError());
        refNode->stop();
        return 1;
    }
    mn::log::info("pair calibration: stand where both sensors see your full body; collecting "
                  "until ",
                  opt.minSamples, " point pairs or ", fmtF(maxSeconds, 0), " s");

    const double tEnd = mn::nowSeconds() + maxSeconds;
    double lastStatus = mn::nowSeconds();
    while (!g_stop.load() && mn::nowSeconds() < tEnd) {
        sleepMs(50);
        size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(mtx);
            n = session.sampleCount();
        }
        if (n >= opt.minSamples)
            break;
        const double now = mn::nowSeconds();
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            mn::log::info("samples: ", n, " / ", opt.minSamples);
        }
    }
    tgtNode->stop();
    refNode->stop();

    const mn::RigidFit fit = session.solve();
    if (!fit.ok) {
        mn::log::error("pair calibration failed (", session.sampleCount(),
                       " point pairs collected); make sure both sensors track the body at the "
                       "same time and move around a little");
        return 1;
    }

    Pose eRef = Pose::identity();
    if (auto stored = store.nodeExtrinsic(refId)) {
        eRef = *stored;
    } else {
        mn::log::info("reference '", refId,
                      "' has no stored extrinsic; it defines the world frame (identity)");
    }
    const Pose eTgt = eRef.compose(fit.transform);
    store.setNodeExtrinsic(tgtId, eTgt);
    if (!store.save(cfg.calibrationFile)) {
        mn::log::error("failed to save calibration file: ", cfg.calibrationFile);
        return 1;
    }
    std::cout << "pair calibration ok: rmse " << fmtF(static_cast<double>(fit.rmse) * 100.0, 2)
              << " cm over " << fit.samples << " point pairs\n";
    std::cout << "saved extrinsic for '" << tgtId << "' to " << cfg.calibrationFile << "\n";
    return 0;
}

// ---------------------------------------------------------- calibrate body

int cmdCalibrateBody(int argc, char** argv, const NodeRegistry& nodeReg) {
    std::string configPath;
    double seconds = 15.0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("calibrate body: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--seconds") {
            if (++i >= argc || !parseDouble(argv[i], seconds) || seconds <= 0.0)
                return usageError("calibrate body: --seconds needs a positive number");
        } else {
            return usageError("calibrate body: unknown option '" + a + "'");
        }
    }
    if (configPath.empty())
        return usageError("calibrate body: -c <config.json> is required");

    AppConfig cfg;
    if (!loadConfig(configPath, cfg))
        return 1;
    CalibrationStore store;
    if (!loadStore(cfg.calibrationFile, store))
        return 1;

    // FusionEngine outlives the nodes (declared first) so capture threads can
    // never touch a dead engine.
    mn::FusionEngine fusion(cfg.fusion);
    std::vector<std::unique_ptr<ICaptureNode>> nodes;
    if (!buildAllNodes(cfg, nodeReg, nodes))
        return 1;
    for (const auto& entry : cfg.nodes)
        fusion.setNode(entry.id, resolveExtrinsic(store, entry));
    // Intentionally no setBodyModel here: the old model must not bias the new
    // estimate through the bone-length constraint.

    std::atomic<uint64_t> framesIn{0};
    FrameCallback cb = [&fusion, &framesIn](const NodeDescriptor& d, const SkeletonFrame& f) {
        framesIn.fetch_add(1, std::memory_order_relaxed);
        fusion.submit(d.id, f);
    };
    if (!startAllNodes(nodes, cb))
        return 1;
    mn::log::info("body calibration: stand upright in view of all sensors for ", fmtF(seconds, 0),
                  " s");

    mn::BodyModelEstimator est;
    const double tEnd = mn::nowSeconds() + seconds;
    double lastStatus = mn::nowSeconds();
    while (!g_stop.load() && mn::nowSeconds() < tEnd) {
        sleepMs(33); // ~30 Hz
        const double now = mn::nowSeconds();
        SkeletonFrame world;
        if (fusion.fuse(now, world))
            est.feed(world);
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            mn::log::info("fused frames ", est.frameCount(), " | frames in ", framesIn.load());
        }
    }
    stopAllNodes(nodes);

    const mn::BodyModel model = est.estimate();
    if (!model.valid) {
        mn::log::error("body calibration failed: not enough fused frames (got ", est.frameCount(),
                       ")");
        return 1;
    }

    std::cout << "bone lengths (" << est.frameCount() << " fused frames):\n";
    char line[128];
    for (size_t i = 0; i < mn::kJointCount; ++i) {
        const Joint j = static_cast<Joint>(i);
        if (j == Joint::Hips)
            continue;
        const float len = model.boneLengthToParent[i];
        if (len <= 0.0f)
            continue;
        std::snprintf(line, sizeof(line), "  %-12s -> %-12s %6.1f cm", mn::jointName(j),
                      mn::jointName(mn::jointParent(j)), static_cast<double>(len) * 100.0);
        std::cout << line << "\n";
    }
    store.setBodyModel(model);
    if (!store.save(cfg.calibrationFile)) {
        mn::log::error("failed to save calibration file: ", cfg.calibrationFile);
        return 1;
    }
    std::cout << "saved body model to " << cfg.calibrationFile << "\n";
    return 0;
}

// ----------------------------------------------------- calibrate playspace

int cmdCalibratePlayspace(int argc, char** argv, const NodeRegistry& nodeReg) {
#ifndef MN_HAS_OPENVR_CLIENT
    (void)argc;
    (void)argv;
    (void)nodeReg;
    mn::log::error("this build has no OpenVR client support; rebuild with "
                   "-DMN_WITH_OPENVR_CLIENT=ON and the openvr_api library available");
    return 1;
#else
    std::string configPath;
    std::string hand = "right";
    double seconds = 20.0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("calibrate playspace: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--hand") {
            if (++i >= argc)
                return usageError("calibrate playspace: --hand needs a value");
            hand = argv[i];
            if (hand != "left" && hand != "right")
                return usageError("calibrate playspace: --hand must be 'left' or 'right'");
        } else if (a == "--seconds") {
            if (++i >= argc || !parseDouble(argv[i], seconds) || seconds <= 0.0)
                return usageError("calibrate playspace: --seconds needs a positive number");
        } else {
            return usageError("calibrate playspace: unknown option '" + a + "'");
        }
    }
    if (configPath.empty())
        return usageError("calibrate playspace: -c <config.json> is required");

    AppConfig cfg;
    if (!loadConfig(configPath, cfg))
        return 1;
    CalibrationStore store;
    if (!loadStore(cfg.calibrationFile, store))
        return 1;

    vr::EVRInitError initErr = vr::VRInitError_None;
    vr::IVRSystem* vrSys = vr::VR_Init(&initErr, vr::VRApplication_Background);
    if (vrSys == nullptr || initErr != vr::VRInitError_None) {
        mn::log::error("OpenVR init failed: ", vr::VR_GetVRInitErrorAsSymbol(initErr), " (",
                       vr::VR_GetVRInitErrorAsEnglishDescription(initErr),
                       "); is SteamVR running?");
        return 1;
    }
    struct VrShutdown {
        ~VrShutdown() { vr::VR_Shutdown(); }
    } vrShutdown;

    mn::FusionEngine fusion(cfg.fusion);
    std::vector<std::unique_ptr<ICaptureNode>> nodes;
    if (!buildAllNodes(cfg, nodeReg, nodes))
        return 1;
    for (const auto& entry : cfg.nodes)
        fusion.setNode(entry.id, resolveExtrinsic(store, entry));
    if (store.bodyModel().valid)
        fusion.setBodyModel(store.bodyModel());

    std::atomic<uint64_t> framesIn{0};
    FrameCallback cb = [&fusion, &framesIn](const NodeDescriptor& d, const SkeletonFrame& f) {
        framesIn.fetch_add(1, std::memory_order_relaxed);
        fusion.submit(d.id, f);
    };
    if (!startAllNodes(nodes, cb))
        return 1;

    const bool left = (hand == "left");
    const Joint wrist = left ? Joint::WristL : Joint::WristR;

    mn::log::info("playspace calibration: hold the ", hand,
                  " controller in that hand and wave it through the play area for ",
                  fmtF(seconds, 0), " s");

    // The shared sampler runs on this thread: `pump` fuses at ~30 Hz and
    // `fusedWrist` reads the result back, so no locking is needed between them.
    SkeletonFrame fusedWorld;
    bool fusedOk = false;
    const auto pump = [&fusion, &fusedWorld, &fusedOk]() {
        fusedOk = fusion.fuse(mn::nowSeconds(), fusedWorld);
    };
    const auto fusedWrist = [&fusedWorld, &fusedOk, wrist]() -> std::optional<Vec3> {
        if (!fusedOk || fusedWorld[wrist].state == mn::TrackState::NotTracked)
            return std::nullopt;
        return fusedWorld[wrist].pos;
    };
    std::atomic<bool> cancel{false}; // Ctrl+C is handled via g_stop inside the loop
    std::string sampleErr;
    const mn::RigidFit fit =
        collectPlayspaceAnchor(vrSys, hand, seconds, fusedWrist, pump, {}, cancel, sampleErr);
    stopAllNodes(nodes);

    if (!fit.ok) {
        mn::log::error(sampleErr);
        return 1;
    }
    store.setWorldAnchor(fit.transform);
    if (!store.save(cfg.calibrationFile)) {
        mn::log::error("failed to save calibration file: ", cfg.calibrationFile);
        return 1;
    }
    std::cout << "playspace anchor ok: rmse " << fmtF(static_cast<double>(fit.rmse) * 100.0, 2)
              << " cm over " << fit.samples << " samples\n";
    std::cout << "saved world anchor to " << cfg.calibrationFile << "\n";
    return 0;
#endif
}

// ------------------------------------------------------------------ doctor

// Keep in sync with the CMake project() version (no shared header carries it).
constexpr const char* kAppVersion = "0.1.0";

const char* platformName() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

// getenv without the MSVC CRT deprecation warning.
std::string envValue(const char* name) {
#ifdef _WIN32
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || buf == nullptr)
        return {};
    std::string v(buf);
    std::free(buf);
    return v;
#else
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
#endif
}

// True when the last path component of `p` is "marionette" (case-insensitive,
// trailing separators ignored) - the shape of our SteamVR external_drivers
// registration.
bool pathEndsWithMarionette(std::string p) {
    while (!p.empty() && (p.back() == '/' || p.back() == '\\'))
        p.pop_back();
    const size_t cut = p.find_last_of("/\\");
    std::string last = (cut == std::string::npos) ? p : p.substr(cut + 1);
    std::transform(last.begin(), last.end(), last.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return last == "marionette";
}

struct SteamVrProbe {
    std::string path;              // expected openvrpaths.vrpath location ("" = unknown)
    bool found = false;            // file exists and is readable
    bool driverRegistered = false; // external_drivers has a path ending in "marionette"
};

// Small local probe of SteamVR's openvrpaths.vrpath (best effort, no OpenVR
// or mn_dashboard dependency). Presence implies SteamVR has run on this
// machine at least once.
SteamVrProbe probeSteamVr() {
    SteamVrProbe r;
#ifdef _WIN32
    const std::string base = envValue("LOCALAPPDATA");
    if (base.empty())
        return r;
    r.path = base + "\\openvr\\openvrpaths.vrpath";
#else
    std::string base = envValue("XDG_CONFIG_HOME");
    if (base.empty()) {
        const std::string home = envValue("HOME");
        if (home.empty())
            return r;
        base = home + "/.config";
    }
    r.path = base + "/openvr/openvrpaths.vrpath";
#endif
    std::ifstream f(r.path);
    if (!f.is_open())
        return r;
    r.found = true;
    try {
        const nlohmann::json j = nlohmann::json::parse(f);
        const auto it = j.find("external_drivers");
        if (it != j.end() && it->is_array()) {
            for (const auto& d : *it) {
                if (d.is_string() && pathEndsWithMarionette(d.get<std::string>())) {
                    r.driverRegistered = true;
                    break;
                }
            }
        }
    } catch (const std::exception&) {
        // Malformed file: report found but not registered (best effort).
    }
    return r;
}

struct SensorProbe {
    std::string label;    // "kinect_v2", "kinect_v1[0]", ...
    int index = -1;       // kinect_v1 enumeration index; -1 for kinect_v2
    bool created = false; // registry factory produced a node
    bool present = false; // start() succeeded
    bool failure = false; // hard failure: broken, not merely absent
    uint64_t frames = 0;
    double seconds = 0.0;
    std::string error;
};

#if defined(MN_HAS_KINECT_V2) || defined(MN_HAS_KINECT_V1)

constexpr double kDoctorProbeSeconds = 3.0;

// Creates one node through the registry and runs it briefly with a counting
// callback. A start() error containing `notPresentNeedle` is the backend's
// "no sensor there" case: INFO on a machine without hardware, not a failure.
SensorProbe probeSensor(const NodeRegistry& reg, const std::string& type,
                        const std::string& label, int index, const nlohmann::json& params,
                        const char* notPresentNeedle) {
    SensorProbe p;
    p.label = label;
    p.index = index;
    std::string err;
    auto node = reg.create(type, "doctor", params, err);
    if (!node) {
        p.error = err;
        p.failure = true;
        return p;
    }
    p.created = true;
    std::atomic<uint64_t> frames{0};
    FrameCallback cb = [&frames](const NodeDescriptor&, const SkeletonFrame&) {
        frames.fetch_add(1, std::memory_order_relaxed);
    };
    if (!node->start(cb)) {
        p.error = node->lastError();
        p.failure = (p.error.find(notPresentNeedle) == std::string::npos);
        return p;
    }
    p.present = true;
    const double tEnd = mn::nowSeconds() + kDoctorProbeSeconds;
    while (!g_stop.load() && mn::nowSeconds() < tEnd)
        sleepMs(50);
    node->stop(); // joins the capture thread; no callbacks after this
    p.frames = frames.load();
    p.seconds = kDoctorProbeSeconds;
    if (p.frames == 0) {
        p.failure = true;
        p.error = "sensor started but delivered no frames (not even keepalives) in " +
                  fmtF(kDoctorProbeSeconds, 0) + " s";
    }
    return p;
}

#endif // MN_HAS_KINECT_V2 || MN_HAS_KINECT_V1

int cmdDoctor(int argc, char** argv, const NodeRegistry& nodeReg, const EndpointRegistry& epReg) {
    std::string configPath;
    bool jsonOut = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" || a == "--config") {
            if (++i >= argc)
                return usageError("doctor: " + a + " needs a value");
            configPath = argv[i];
        } else if (a == "--json") {
            jsonOut = true;
        } else {
            return usageError("doctor: unknown option '" + a + "'");
        }
    }
    if (jsonOut)
        mn::log::setLevel(mn::log::Level::Error); // keep stdout machine-readable

    std::vector<std::string> problems;

    // --- compiled backends + SDK environment -------------------------------
#ifdef MN_HAS_KINECT_V2
    const bool hasV2 = true;
#else
    const bool hasV2 = false;
#endif
#ifdef MN_HAS_KINECT_V1
    const bool hasV1 = true;
#else
    const bool hasV1 = false;
#endif
#ifdef MN_HAS_OPENVR_CLIENT
    const bool hasOvr = true;
#else
    const bool hasOvr = false;
#endif
#ifdef MN_HAS_DASHBOARD
    const bool hasDash = true;
#else
    const bool hasDash = false;
#endif
    const std::string sdk20 = envValue("KINECTSDK20_DIR");
    const std::string sdk10 = envValue("KINECTSDK10_DIR");

    // --- sensor probes ------------------------------------------------------
    // Backend nodes log expected absence ("not present") at warn/error level;
    // silence the console during probes so the report stays clean. The
    // EventLog sink still records everything.
    std::vector<SensorProbe> probes;
    const mn::log::Level probeLogPrev = mn::log::level();
    mn::log::setLevel(mn::log::Level::Off);
    if (!jsonOut && (hasV2 || hasV1))
        std::cout << "probing sensors (~3 s per sensor)...\n";
#ifdef MN_HAS_KINECT_V2
    probes.push_back(probeSensor(nodeReg, "kinect_v2", "kinect_v2", -1, nlohmann::json::object(),
                                 "no Kinect v2 sensor present"));
#endif
#ifdef MN_HAS_KINECT_V1
    for (int idx = 0; idx <= 3; ++idx) {
        SensorProbe p = probeSensor(nodeReg, "kinect_v1", "kinect_v1[" + std::to_string(idx) + "]",
                                    idx, nlohmann::json{{"index", idx}}, "out of range");
        // Stop on create failure or on the first absent index (contiguous
        // enumeration); keep going past a broken-but-present sensor.
        const bool stopProbing = !p.created || (!p.present && !p.failure);
        probes.push_back(std::move(p));
        if (stopProbing)
            break;
    }
#endif
    mn::log::setLevel(probeLogPrev);
    for (const auto& p : probes) {
        if (p.failure)
            problems.push_back("sensor " + p.label + ": " + p.error);
    }

    // --- config check (-c) --------------------------------------------------
    AppConfig cfg; // defaults double as the no-config fallback below
    const bool cfgGiven = !configPath.empty();
    bool cfgOk = false;
    std::string cfgError;
    if (cfgGiven) {
        try {
            cfg = AppConfig::load(configPath);
            cfgOk = true;
            const auto nodeTypes = nodeReg.types();
            for (const auto& n : cfg.nodes) {
                if (std::find(nodeTypes.begin(), nodeTypes.end(), n.type) == nodeTypes.end())
                    problems.push_back("config: node '" + n.id + "' uses type '" + n.type +
                                       "' which is not registered in this build");
            }
            const auto epTypes = epReg.types();
            for (const auto& e : cfg.endpoints) {
                if (std::find(epTypes.begin(), epTypes.end(), e.type) == epTypes.end())
                    problems.push_back("config: endpoint type '" + e.type +
                                       "' is not registered in this build");
            }
        } catch (const std::exception& e) {
            cfgError = e.what();
            problems.push_back("config: failed to load '" + configPath + "': " + cfgError);
        }
    }

    // --- calibration store summary -------------------------------------------
    const std::string calibPath = cfg.calibrationFile;
    CalibrationStore store;
    const bool calibOk = store.load(calibPath);
    bool calibExists = false;
    {
        std::ifstream f(calibPath);
        calibExists = f.is_open();
    }
    if (!calibOk)
        problems.push_back("calibration: malformed file: " + calibPath);
    // CalibrationStore cannot enumerate extrinsic ids; peek at the JSON itself.
    std::vector<std::string> extrinsicIds;
    if (calibExists && calibOk) {
        try {
            std::ifstream f(calibPath);
            const nlohmann::json cj = nlohmann::json::parse(f);
            if (const auto it = cj.find("extrinsics"); it != cj.end() && it->is_object()) {
                for (auto e = it->begin(); e != it->end(); ++e)
                    extrinsicIds.push_back(e.key());
            }
        } catch (const std::exception&) {
            // store.load() accepted the file; enumeration stays best effort.
        }
    }
    const bool bodyValid = store.bodyModel().valid;
    const bool anchorSet = store.worldAnchor().has_value();

    // --- ports ----------------------------------------------------------------
    mn::UdpSocket::globalInit();
    bool wireFree = false;
    std::string wireError;
    {
        mn::UdpSocket s;
        wireFree = s.openReceive(mn::wire::kDefaultPort, "127.0.0.1");
        if (!wireFree)
            wireError = s.lastError();
    } // socket closes here, releasing the port

    // --- SteamVR ----------------------------------------------------------------
    const SteamVrProbe svr = probeSteamVr();

    // --- report -----------------------------------------------------------------
    if (jsonOut) {
        nlohmann::json j;
        j["app"] = {{"name", "marionette"}, {"version", kAppVersion}};
        j["platform"] = platformName();
        j["backends"] = {{"kinect_v2", hasV2},
                         {"kinect_v1", hasV1},
                         {"openvr_client", hasOvr},
                         {"dashboard", hasDash}};
        j["sdk_env"] = {{"KINECTSDK20_DIR", sdk20}, {"KINECTSDK10_DIR", sdk10}};
        nlohmann::json sensors = nlohmann::json::array();
        for (const auto& p : probes) {
            nlohmann::json s;
            s["label"] = p.label;
            if (p.index >= 0)
                s["index"] = p.index;
            s["present"] = p.present;
            s["failure"] = p.failure;
            s["frames"] = p.frames;
            s["seconds"] = p.seconds;
            s["error"] = p.error;
            sensors.push_back(std::move(s));
        }
        j["sensors"] = std::move(sensors);
        if (cfgGiven) {
            j["config"] = {{"path", configPath},
                           {"ok", cfgOk},
                           {"error", cfgError},
                           {"nodes", cfg.nodes.size()},
                           {"endpoints", cfg.endpoints.size()}};
        }
        j["calibration"] = {{"file", calibPath},
                            {"exists", calibExists},
                            {"ok", calibOk},
                            {"extrinsics", extrinsicIds},
                            {"body_model_valid", bodyValid},
                            {"world_anchor_set", anchorSet}};
        j["ports"] = {{"wire", mn::wire::kDefaultPort},
                      {"wire_udp_free", wireFree},
                      {"wire_error", wireError},
                      {"dashboard", cfg.dashboard.port}};
        j["steamvr"] = {{"openvrpaths_found", svr.found},
                        {"openvrpaths_path", svr.path},
                        {"driver_registered", svr.driverRegistered}};
        j["problems"] = problems;
        j["ok"] = problems.empty();
        std::cout << j.dump(2) << "\n";
    } else {
        const auto row = [](const std::string& k, const std::string& v) {
            std::cout << "  " << k << ' ';
            for (size_t i = k.size() + 1; i < 21; ++i)
                std::cout << ' ';
            std::cout << v << "\n";
        };
        const auto compiled = [](bool b) {
            return b ? std::string("compiled") : std::string("not compiled");
        };
        std::cout << "\nmarionette doctor\n";
        row("version", kAppVersion);
        row("platform", platformName());
        std::cout << "\nbackends\n";
        row("kinect_v2", compiled(hasV2));
        row("kinect_v1", compiled(hasV1));
        row("openvr_client", compiled(hasOvr));
        row("dashboard", compiled(hasDash));
        row("KINECTSDK20_DIR", sdk20.empty() ? "(not set)" : sdk20);
        row("KINECTSDK10_DIR", sdk10.empty() ? "(not set)" : sdk10);
        std::cout << "\nsensors\n";
        if (probes.empty())
            row("(none)", "no Kinect backend compiled into this build");
        for (const auto& p : probes) {
            if (p.present && !p.failure) {
                row(p.label, "OK: " + std::to_string(p.frames) + " frames in " +
                                 fmtF(p.seconds, 1) + " s (" +
                                 fmtF(static_cast<double>(p.frames) / p.seconds, 1) + " fps)");
            } else if (p.failure) {
                row(p.label, "FAIL: " + p.error);
            } else {
                row(p.label, "not present");
            }
        }
        std::cout << "\nconfig\n";
        if (!cfgGiven) {
            row("file", "(none given; pass -c <config.json> to check one)");
        } else if (cfgOk) {
            row("file", configPath);
            row("load", "OK: " + std::to_string(cfg.nodes.size()) + " node(s), " +
                            std::to_string(cfg.endpoints.size()) + " endpoint(s)");
        } else {
            row("file", configPath);
            row("load", "FAIL: " + cfgError);
        }
        std::cout << "\ncalibration\n";
        row("file", calibPath + (calibExists ? "" : " (missing; empty store)"));
        if (!calibOk) {
            row("load", "FAIL: malformed");
        } else {
            std::string ids;
            for (const auto& id : extrinsicIds) {
                if (!ids.empty())
                    ids += ", ";
                ids += id;
            }
            row("extrinsics",
                std::to_string(extrinsicIds.size()) + (ids.empty() ? "" : " (" + ids + ")"));
            row("body model", bodyValid ? "valid" : "not calibrated");
            row("world anchor", anchorSet ? "set" : "not set");
        }
        std::cout << "\nports\n";
        row("udp " + std::to_string(mn::wire::kDefaultPort) + " (wire)",
            wireFree ? std::string("free")
                     : "in use (SteamVR driver listening, or another app): " + wireError);
        row("dashboard", std::to_string(cfg.dashboard.port) + " (configured; not probed)");
        std::cout << "\nsteamvr\n";
        row("openvrpaths", svr.found ? "found: " + svr.path
                                     : "not found" + (svr.path.empty() ? std::string()
                                                                       : ": " + svr.path));
        row("driver registered", svr.driverRegistered ? "yes" : "no");
        std::cout << "\n";
        if (problems.empty()) {
            std::cout << "result: OK\n";
        } else {
            std::cout << "result: " << problems.size() << " problem(s)\n";
            for (const auto& p : problems)
                std::cout << "  - " << p << "\n";
        }
    }
    return problems.empty() ? 0 : 1;
}

} // namespace

// -------------------------------------------------------------------- main

int main(int argc, char** argv) {
    // Route every log line into the EventLog ring buffer (dashboard /api/events
    // and doctor read it) before anything can log.
    mn::EventLog::installLogCapture();
    std::signal(SIGINT, &mnOnSignal);
    std::signal(SIGTERM, &mnOnSignal);

    NodeRegistry nodeReg;
    EndpointRegistry epReg;
    mn::mock::registerNodes(nodeReg);
#ifdef MN_HAS_KINECT_V2
    mn::kinect2::registerNodes(nodeReg);
#endif
#ifdef MN_HAS_KINECT_V1
    mn::kinect1::registerNodes(nodeReg);
#endif
    mn::osc::registerEndpoints(epReg);
    mn::ovrbridge::registerEndpoints(epReg);

    if (argc < 2) {
        printUsage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            printUsage();
            return 0;
        }
        if (cmd == "list-types")
            return cmdListTypes(nodeReg, epReg);
        if (cmd == "run")
            return cmdRun(argc, argv, nodeReg, epReg);
        if (cmd == "record")
            return cmdRecord(argc, argv, nodeReg);
        if (cmd == "doctor")
            return cmdDoctor(argc, argv, nodeReg, epReg);
        if (cmd == "calibrate") {
            if (argc < 3)
                return usageError("calibrate: expected a mode (pair | body | playspace)");
            const std::string mode = argv[2];
            if (mode == "pair")
                return cmdCalibratePair(argc, argv, nodeReg);
            if (mode == "body")
                return cmdCalibrateBody(argc, argv, nodeReg);
            if (mode == "playspace")
                return cmdCalibratePlayspace(argc, argv, nodeReg);
            return usageError("calibrate: unknown mode '" + mode + "'");
        }
        return usageError("unknown command '" + cmd + "'");
    } catch (const std::exception& e) {
        mn::log::error("fatal: ", e.what());
        return 1;
    }
}
