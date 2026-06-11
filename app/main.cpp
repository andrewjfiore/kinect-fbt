// Marionette command line application.
//
// Subcommands:
//   list-types            print capture node and endpoint types in this build
//   run                   full pipeline: capture -> fusion -> mapping -> endpoints
//   record                run one capture node, dump its frames to JSONL
//   calibrate pair        solve a target sensor's extrinsic against a reference sensor
//   calibrate body        estimate per-user bone lengths from fused frames
//   calibrate playspace   anchor the Marionette world to the SteamVR playspace
//
// Exit codes: 0 success, 1 runtime failure, 2 usage error.

#include "mn/calibration.hpp"
#include "mn/capture.hpp"
#include "mn/clock.hpp"
#include "mn/config.hpp"
#include "mn/endpoint.hpp"
#include "mn/fusion.hpp"
#include "mn/log.hpp"
#include "mn/pipeline.hpp"
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

#ifdef MN_HAS_OPENVR_CLIENT
#include <openvr.h>
#endif

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};

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
           "  marionette record -c <config.json> --node <id> -o <out.jsonl> [--duration <sec>]\n"
           "  marionette calibrate pair -c <config.json> --reference <id> --target <id>\n"
           "                            [--min-samples <n>] [--max-seconds <sec>]\n"
           "  marionette calibrate body -c <config.json> [--seconds <sec>]\n"
           "  marionette calibrate playspace -c <config.json> [--hand left|right]\n"
           "                                 [--seconds <sec>]\n"
           "  marionette -h | --help\n"
           "\n"
           "Commands:\n"
           "  list-types           List node and endpoint types compiled into this build.\n"
           "  run                  Run the pipeline: capture -> fusion -> trackers -> endpoints.\n"
           "  record               Run a single capture node, write its frames to a JSONL file.\n"
           "  calibrate pair       Solve a target sensor's extrinsic against a reference sensor.\n"
           "  calibrate body       Estimate per-user bone lengths from fused frames.\n"
           "  calibrate playspace  Anchor the Marionette world to the SteamVR playspace.\n"
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
    double duration = 0.0; // 0 = run until interrupted
    bool verbose = false;
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
    const vr::ETrackedControllerRole role =
        left ? vr::TrackedControllerRole_LeftHand : vr::TrackedControllerRole_RightHand;

    mn::log::info("playspace calibration: hold the ", hand,
                  " controller in that hand and wave it through the play area for ",
                  fmtF(seconds, 0), " s");

    std::vector<Vec3> worldPts, steamvrPts;
    bool warnedNoController = false;
    const double tEnd = mn::nowSeconds() + seconds;
    double lastStatus = mn::nowSeconds();
    int tick = 0;
    while (!g_stop.load() && mn::nowSeconds() < tEnd) {
        sleepMs(33); // fuse at ~30 Hz; sample pairs on every 3rd tick (~10 Hz)
        const double now = mn::nowSeconds();
        SkeletonFrame world;
        const bool fused = fusion.fuse(now, world);
        if (now - lastStatus >= 1.0) {
            lastStatus = now;
            mn::log::info("anchor samples: ", worldPts.size());
        }
        ++tick;
        if (tick % 3 != 0)
            continue;
        if (!fused || world[wrist].state == mn::TrackState::NotTracked)
            continue;
        const vr::TrackedDeviceIndex_t idx = vrSys->GetTrackedDeviceIndexForControllerRole(role);
        if (idx == vr::k_unTrackedDeviceIndexInvalid || idx >= vr::k_unMaxTrackedDeviceCount) {
            if (!warnedNoController) {
                warnedNoController = true;
                mn::log::warn("no ", hand, " controller is tracked yet; waiting for it");
            }
            continue;
        }
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
        vrSys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, poses,
                                               vr::k_unMaxTrackedDeviceCount);
        const vr::TrackedDevicePose_t& devPose = poses[idx];
        if (!devPose.bPoseIsValid)
            continue;
        const auto& m = devPose.mDeviceToAbsoluteTracking.m;
        const Vec3 vrPos(m[0][3], m[1][3], m[2][3]);
        // Skip near-duplicate points so a resting controller cannot dominate the fit.
        if (!steamvrPts.empty() && (vrPos - steamvrPts.back()).norm() < 0.01f)
            continue;
        worldPts.push_back(world[wrist].pos);
        steamvrPts.push_back(vrPos);
    }
    stopAllNodes(nodes);

    const mn::RigidFit fit = mn::solveAnchor(worldPts, steamvrPts);
    if (!fit.ok) {
        mn::log::error("playspace calibration failed (", worldPts.size(),
                       " samples); wave the controller through a larger, non-flat volume while "
                       "the sensors track you");
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

} // namespace

// -------------------------------------------------------------------- main

int main(int argc, char** argv) {
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
