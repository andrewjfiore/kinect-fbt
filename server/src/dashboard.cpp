// Dashboard HTTP server: serves the embedded single-file UI plus the JSON API
// documented in docs/DASHBOARD_API.md, and runs calibration jobs (pair / body /
// playspace) on a single worker thread using the pipeline's observer hooks.

// httplib first: it pulls winsock2.h before anything can include windows.h.
#include <httplib.h>

#include "web_embedded.hpp"

#include "mn_dashboard/dashboard.hpp"

#include "mn/calibration.hpp"
#include "mn/clock.hpp"
#include "mn/config.hpp"
#include "mn/events.hpp"
#include "mn/fusion.hpp"
#include "mn/log.hpp"
#include "mn/mapping.hpp"
#include "mn/pipeline.hpp"
#include "mn/skeleton.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mn::dash {

namespace {

using nlohmann::json;

bool readFile(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

std::string envVar(const char* name) {
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
    return v ? std::string(v) : std::string();
#endif
}

// Last path component equals "marionette" (case-insensitive, ignoring
// trailing separators).
bool pathEndsWithMarionette(const std::string& raw) {
    std::string p = raw;
    while (!p.empty() && (p.back() == '/' || p.back() == '\\'))
        p.pop_back();
    const size_t pos = p.find_last_of("/\\");
    std::string last = (pos == std::string::npos) ? p : p.substr(pos + 1);
    std::transform(last.begin(), last.end(), last.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return last == "marionette";
}

const char* levelName(log::Level lvl) {
    switch (lvl) {
    case log::Level::Debug:
        return "debug";
    case log::Level::Info:
        return "info";
    case log::Level::Warn:
        return "warn";
    case log::Level::Error:
        return "error";
    }
    return "info";
}

json vec3Json(const Vec3& v) {
    return json::array({v.x(), v.y(), v.z()});
}

json quatJson(const Quat& q) {
    return json::array({q.w(), q.x(), q.y(), q.z()});
}

std::string fmt2(double v) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

void sendJson(httplib::Response& res, const json& j, int status = 200) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(j.dump(), "application/json");
}

void sendError(httplib::Response& res, int status, const std::string& msg) {
    sendJson(res, json{{"ok", false}, {"error", msg}}, status);
}

void sendOk(httplib::Response& res) {
    sendJson(res, json{{"ok", true}, {"error", ""}});
}

// Empty body -> empty object (all calibration POST bodies are optional except
// pair's reference/target, validated by the handler).
bool parseBody(const httplib::Request& req, json& out, std::string& err) {
    if (req.body.empty()) {
        out = json::object();
        return true;
    }
    out = json::parse(req.body, nullptr, false);
    if (out.is_discarded() || !out.is_object()) {
        err = "body must be a JSON object";
        return false;
    }
    return true;
}

std::string jsonString(const json& j, const char* key) {
    if (j.contains(key) && j.at(key).is_string())
        return j.at(key).get<std::string>();
    return {};
}

bool jsonNumber(const json& j, const char* key, double& out) {
    if (!j.contains(key))
        return true; // keep default
    if (!j.at(key).is_number())
        return false;
    out = j.at(key).get<double>();
    return true;
}

// Probe SteamVR's openvrpaths.vrpath: found = a candidate file was readable;
// registered = its external_drivers lists a path ending in "marionette"
// (best effort; stays false on parse failure).
void probeOpenvrPaths(bool& found, bool& registered) {
    found = false;
    registered = false;
    std::vector<std::string> candidates;
#ifdef _WIN32
    const std::string localAppData = envVar("LOCALAPPDATA");
    if (!localAppData.empty())
        candidates.push_back(localAppData + "\\openvr\\openvrpaths.vrpath");
#else
    const std::string home = envVar("HOME");
    if (!home.empty()) {
        candidates.push_back(home + "/.config/openvr/openvrpaths.vrpath");
        candidates.push_back(home + "/.steam/steam/config/openvrpaths.vrpath");
    }
#endif
    for (const auto& path : candidates) {
        std::string text;
        if (!readFile(path, text))
            continue;
        found = true;
        const json j = json::parse(text, nullptr, false);
        if (j.is_discarded() || !j.is_object() || !j.contains("external_drivers"))
            continue;
        const json& drivers = j.at("external_drivers");
        if (!drivers.is_array())
            continue;
        for (const auto& d : drivers) {
            if (d.is_string() && pathEndsWithMarionette(d.get<std::string>())) {
                registered = true;
                return;
            }
        }
    }
}

struct JobState {
    bool active = false;
    std::string kind;  // "pair" | "body" | "playspace" | "" before any job
    std::string phase; // job-defined short string
    size_t done = 0;
    size_t needed = 0;
    bool finished = false; // a result is available (until the next job starts)
    bool ok = false;
    double rmseCm = 0.0;
    std::string message;
};

} // namespace

// ---------------------------------------------------------------------------

struct DashboardServer::Impl {
    Pipeline& pipeline;
    CalibrationStore& store;
    const std::string calibPath;
    const Options opt;

    httplib::Server svr;
    std::thread serverThread;
    std::atomic<bool> running{false};
    std::string lastError;
    double startTime = 0.0;

    // Guards `store` against concurrent access from server threads (status /
    // skeleton / doctor reads) and the calibration worker (writes + save).
    std::mutex storeMutex;

    std::mutex jobMutex; // guards `job`
    JobState job;
    std::thread jobThread;
    std::atomic<bool> jobBusy{false};
    std::atomic<bool> jobCancel{false};

    Impl(Pipeline& p, CalibrationStore& s, std::string path, Options o)
        : pipeline(p), store(s), calibPath(std::move(path)), opt(std::move(o)) {
        setupRoutes();
    }

    ~Impl() { stopImpl(); }

    // ------------------------------------------------------------ lifecycle

    bool startImpl() {
        if (running.load())
            return true;
        if (!svr.bind_to_port(opt.bind, opt.port)) {
            lastError = "failed to bind " + opt.bind + ":" + std::to_string(opt.port) +
                        " (port already in use?)";
            return false;
        }
        startTime = nowSeconds(); // before the first request can read uptime
        serverThread = std::thread([this] { svr.listen_after_bind(); });
        // Wait until the accept loop is live so a racing stop() always sees a
        // stoppable server.
        const double t0 = nowSeconds();
        while (!svr.is_running() && nowSeconds() - t0 < 5.0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!svr.is_running()) {
            lastError = "server failed to start listening on " + opt.bind + ":" +
                        std::to_string(opt.port);
            svr.stop();
            if (serverThread.joinable())
                serverThread.join();
            return false;
        }
        running.store(true);
        return true;
    }

    void stopImpl() {
        // Server first: after the join no handler can start a new job.
        if (serverThread.joinable()) {
            svr.stop();
            serverThread.join();
        }
        jobCancel.store(true);
        if (jobThread.joinable())
            jobThread.join();
        running.store(false);
    }

    // ----------------------------------------------------------- job state

    void setProgress(size_t done, size_t needed) {
        std::lock_guard<std::mutex> lk(jobMutex);
        job.done = done;
        job.needed = needed;
    }

    void finishJob(bool ok, double rmseCm, const std::string& message) {
        std::lock_guard<std::mutex> lk(jobMutex);
        job.active = false;
        job.finished = true;
        job.ok = ok;
        job.rmseCm = rmseCm;
        job.message = message;
        job.phase = ok ? "done" : "failed";
    }

    // False when another job is running. On success the worker owns jobBusy
    // and clears it as its very last action.
    bool beginJob(const std::string& kind, const std::string& phase, size_t needed,
                  std::function<void()> fn) {
        bool expected = false;
        if (!jobBusy.compare_exchange_strong(expected, true))
            return false;
        if (jobThread.joinable())
            jobThread.join(); // previous worker already cleared jobBusy
        jobCancel.store(false);
        {
            std::lock_guard<std::mutex> lk(jobMutex);
            job = JobState{};
            job.active = true;
            job.kind = kind;
            job.phase = phase;
            job.needed = needed;
        }
        jobThread = std::thread([this, fn = std::move(fn)] {
            fn();
            jobBusy.store(false);
        });
        return true;
    }

    // ------------------------------------------------------------- helpers

    bool hasNode(const std::string& id) const {
        for (const auto& n : pipeline.appConfig().nodes) {
            if (n.id == id)
                return true;
        }
        return false;
    }

    // Same resolution order as the pipeline: store > params["extrinsic"] >
    // identity.
    Pose resolveExtrinsic(const std::string& id) {
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            if (const auto stored = store.nodeExtrinsic(id))
                return *stored;
        }
        for (const auto& n : pipeline.appConfig().nodes) {
            if (n.id != id)
                continue;
            if (n.params.is_object() && n.params.contains("extrinsic")) {
                try {
                    return n.params.at("extrinsic").get<Pose>();
                } catch (const std::exception&) {
                    break;
                }
            }
            break;
        }
        return Pose::identity();
    }

    // -------------------------------------------------------- JSON builders

    json statusJson() {
        const AppConfig& cfg = pipeline.appConfig();
        const Pipeline::Stats st = pipeline.stats();

        json caps = json::object();
        for (const auto& [k, v] : opt.capabilities)
            caps[k] = v;
        caps["dashboard"] = true;

        const json app{{"name", "marionette"},
                       {"version", "0.1.0"},
                       {"uptime_s", nowSeconds() - startTime},
                       {"tick_hz", cfg.tickHz},
                       {"capabilities", caps}};

        const double pct = (st.ticks > 0) ? 100.0 * static_cast<double>(st.fusedTicks) /
                                                static_cast<double>(st.ticks)
                                          : 0.0;
        const json pipe{{"running", pipeline.isRunning()}, {"frames_in", st.framesIn},
                        {"ticks", st.ticks},               {"fused_ticks", st.fusedTicks},
                        {"fused_pct", pct},                {"trackers_out", st.trackersOut}};

        json nodes = json::array();
        for (const auto& ns : pipeline.nodeStatuses()) {
            nodes.push_back(json{{"id", ns.id},
                                 {"type", ns.type},
                                 {"running", ns.running},
                                 {"frames", ns.frames},
                                 {"fps", ns.fps},
                                 {"last_frame_age_s", ns.lastFrameAge},
                                 {"has_body", ns.lastHasBody},
                                 {"restarts", ns.restarts},
                                 {"last_error", ns.lastError}});
        }

        json eps = json::array();
        for (const auto& e : cfg.endpoints)
            eps.push_back(json{{"type", e.type}});

        json trackers = json::array();
        for (const auto& t : pipeline.latestTrackers()) {
            trackers.push_back(json{{"role", trackerRoleName(t.role)},
                                    {"valid", t.valid},
                                    {"pos", vec3Json(t.pose.pos)}});
        }

        json extr = json::array();
        bool bodyValid = false;
        bool anchorSet = false;
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            for (const auto& n : cfg.nodes) {
                if (store.nodeExtrinsic(n.id))
                    extr.push_back(n.id);
            }
            bodyValid = store.bodyModel().valid;
            anchorSet = store.worldAnchor().has_value();
        }
        const json calib{{"file", calibPath},
                         {"extrinsics", extr},
                         {"body_model_valid", bodyValid},
                         {"world_anchor_set", anchorSet}};

        const auto counts = EventLog::instance().levelCounts();
        const json events{{"warn", counts[2]},
                          {"error", counts[3]},
                          {"latest_seq", EventLog::instance().latestSeq()}};

        return json{{"app", app},           {"pipeline", pipe},    {"nodes", nodes},
                    {"endpoints", eps},     {"trackers", trackers}, {"calibration", calib},
                    {"events", events}};
    }

    json skeletonJson() {
        const SkeletonFrame f = pipeline.latestFused();

        json joints = json::object();
        if (f.hasBody) {
            for (size_t i = 0; i < kJointCount; ++i) {
                const JointSample& s = f.joints[i];
                joints[jointName(static_cast<Joint>(i))] =
                    json{{"p", vec3Json(s.pos)},
                         {"c", s.confidence},
                         {"s", static_cast<int>(s.state)}};
            }
        }

        json bones = json::array();
        for (size_t i = 0; i < kJointCount; ++i) {
            const Joint j = static_cast<Joint>(i);
            if (j == Joint::Hips)
                continue; // root has no parent bone
            bones.push_back(json::array({jointName(j), jointName(jointParent(j))}));
        }

        json sensors = json::array();
        for (const auto& n : pipeline.appConfig().nodes) {
            const Pose e = resolveExtrinsic(n.id);
            const Vec3 fwd = e.rot * Vec3(0.0f, 0.0f, 1.0f);
            sensors.push_back(json{{"id", n.id},
                                   {"type", n.type},
                                   {"pos", vec3Json(e.pos)},
                                   {"fwd", vec3Json(fwd)}});
        }

        json trackers = json::array();
        for (const auto& t : pipeline.latestTrackers()) {
            trackers.push_back(json{{"role", trackerRoleName(t.role)},
                                    {"valid", t.valid},
                                    {"pos", vec3Json(t.pose.pos)},
                                    {"rot", quatJson(t.pose.rot)}});
        }

        return json{{"t", f.timestamp},   {"has_body", f.hasBody}, {"joints", joints},
                    {"bones", bones},     {"sensors", sensors},    {"trackers", trackers}};
    }

    json eventsJson(const httplib::Request& req) {
        uint64_t after = 0;
        if (req.has_param("after")) {
            const std::string v = req.get_param_value("after");
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(v.c_str(), &end, 10);
            if (end != nullptr && *end == '\0')
                after = static_cast<uint64_t>(parsed);
        }

        json arr = json::array();
        for (const Event& e : EventLog::instance().since(after, 500)) {
            arr.push_back(json{{"seq", e.seq},
                               {"t", e.t},
                               {"level", levelName(e.level)},
                               {"msg", e.message}});
        }
        const auto counts = EventLog::instance().levelCounts();
        return json{{"events", arr},
                    {"counts",
                     {{"debug", counts[0]},
                      {"info", counts[1]},
                      {"warn", counts[2]},
                      {"error", counts[3]}}},
                    {"latest_seq", EventLog::instance().latestSeq()}};
    }

    json calibrateStatusJson() {
        std::lock_guard<std::mutex> lk(jobMutex);
        return json{{"active", job.active},
                    {"kind", job.kind},
                    {"phase", job.phase},
                    {"progress", {{"done", job.done}, {"needed", job.needed}}},
                    {"done", job.finished},
                    {"ok", job.ok},
                    {"rmse_cm", job.rmseCm},
                    {"message", job.message}};
    }

    json doctorJson() {
        const AppConfig& cfg = pipeline.appConfig();

#if defined(_WIN32)
        const char* platform = "windows";
#elif defined(__APPLE__)
        const char* platform = "macos";
#else
        const char* platform = "linux";
#endif

        json backends = json::object();
        bool openvrClient = false;
        for (const auto& [k, v] : opt.capabilities) {
            if (k == "openvr_client")
                openvrClient = v;
            else if (k != "dashboard")
                backends[k] = v;
        }

        bool found = false;
        bool registered = false;
        probeOpenvrPaths(found, registered);

        int wirePort = 24190;
        for (const auto& e : cfg.endpoints) {
            if (e.type != "openvr")
                continue;
            if (e.params.is_object() && e.params.contains("port") &&
                e.params.at("port").is_number())
                wirePort = e.params.at("port").get<int>();
            break;
        }

        size_t extrinsics = 0;
        bool bodyValid = false;
        bool anchorSet = false;
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            for (const auto& n : cfg.nodes) {
                if (store.nodeExtrinsic(n.id))
                    ++extrinsics;
            }
            bodyValid = store.bodyModel().valid;
            anchorSet = store.worldAnchor().has_value();
        }

        return json{{"platform", platform},
                    {"backends", backends},
                    {"openvr_client", openvrClient},
                    {"steamvr",
                     {{"openvrpaths_found", found}, {"driver_registered", registered}}},
                    {"ports", {{"dashboard", cfg.dashboard.port}, {"wire", wirePort}}},
                    {"calibration",
                     {{"extrinsics", extrinsics},
                      {"body_model_valid", bodyValid},
                      {"world_anchor_set", anchorSet}}}};
    }

    // ------------------------------------------------------ job worker fns

    void runPairJob(const std::string& refId, const std::string& tgtId, size_t minSamples,
                    double maxSeconds) {
        // Shared with the raw-frame observer; shared_ptr keeps it alive even
        // if a capture-thread callback is still in flight when we clear the
        // observer slot below.
        struct Shared {
            std::mutex m;
            PairCalibrationSession session;
            std::optional<SkeletonFrame> latestRef;
            explicit Shared(const PairCalibrationOptions& o) : session(o) {}
        };
        PairCalibrationOptions sopt;
        sopt.minSamples = minSamples;
        auto shared = std::make_shared<Shared>(sopt);

        // Take the pipeline's raw-frame observer slot for the duration.
        pipeline.setRawFrameObserver(
            [shared, refId, tgtId](const NodeDescriptor& d, const SkeletonFrame& f) {
                std::lock_guard<std::mutex> lk(shared->m);
                if (d.id == refId)
                    shared->latestRef = f;
                else if (d.id == tgtId && shared->latestRef)
                    shared->session.addFramePair(*shared->latestRef, f);
            });

        size_t samples = 0;
        const double tEnd = nowSeconds() + maxSeconds;
        while (!jobCancel.load() && nowSeconds() < tEnd) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            {
                std::lock_guard<std::mutex> lk(shared->m);
                samples = shared->session.sampleCount();
            }
            setProgress(samples, minSamples);
            if (samples >= minSamples)
                break;
        }
        pipeline.setRawFrameObserver({}); // restore the empty slot

        if (jobCancel.load()) {
            finishJob(false, 0.0, "cancelled");
            return;
        }

        RigidFit fit;
        {
            std::lock_guard<std::mutex> lk(shared->m);
            fit = shared->session.solve();
        }
        if (!fit.ok) {
            finishJob(false, 0.0,
                      "pair calibration failed (" + std::to_string(samples) +
                          " matched point pairs); make sure both sensors track the body at "
                          "the same time");
            return;
        }

        const Pose eRef = resolveExtrinsic(refId);
        const Pose eTgt = eRef.compose(fit.transform);
        bool saved = false;
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            store.setNodeExtrinsic(tgtId, eTgt);
            saved = store.save(calibPath);
        }
        if (!pipeline.applyNodeExtrinsic(tgtId, eTgt))
            log::warn("dashboard: applyNodeExtrinsic failed for node '", tgtId, "'");

        const double rmseCm = static_cast<double>(fit.rmse) * 100.0;
        if (!saved) {
            finishJob(false, rmseCm,
                      "extrinsic solved and applied live, but saving failed: " + calibPath);
            return;
        }
        finishJob(true, rmseCm,
                  "extrinsic for '" + tgtId + "' solved from " + std::to_string(fit.samples) +
                      " point pairs (rmse " + fmt2(rmseCm) + " cm); saved and applied live");
    }

    void runBodyJob(double seconds) {
        struct Shared {
            std::mutex m;
            BodyModelEstimator est;
        };
        auto shared = std::make_shared<Shared>();

        // Clear the live body model so the old bone lengths cannot bias the
        // new estimate through the bone-length constraint.
        pipeline.fusion().setBodyModel(BodyModel{});

        pipeline.setFusedFrameObserver(
            [shared](const SkeletonFrame& world, const std::vector<TrackerPose>&) {
                std::lock_guard<std::mutex> lk(shared->m);
                shared->est.feed(world);
            });

        const size_t total = std::max<size_t>(static_cast<size_t>(seconds + 0.5), 1);
        const double t0 = nowSeconds();
        while (!jobCancel.load()) {
            const double elapsed = nowSeconds() - t0;
            if (elapsed >= seconds)
                break;
            setProgress(static_cast<size_t>(elapsed), total);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        pipeline.setFusedFrameObserver({}); // restore the empty slot

        BodyModel model;
        size_t frames = 0;
        {
            std::lock_guard<std::mutex> lk(shared->m);
            model = shared->est.estimate();
            frames = shared->est.frameCount();
        }

        const auto restorePrevious = [this] {
            BodyModel previous;
            {
                std::lock_guard<std::mutex> lk(storeMutex);
                previous = store.bodyModel();
            }
            pipeline.fusion().setBodyModel(previous);
        };

        if (jobCancel.load()) {
            restorePrevious();
            finishJob(false, 0.0, "cancelled");
            return;
        }
        if (!model.valid) {
            restorePrevious();
            finishJob(false, 0.0,
                      "body calibration failed: not enough fused frames (got " +
                          std::to_string(frames) + ")");
            return;
        }

        bool saved = false;
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            store.setBodyModel(model);
            saved = store.save(calibPath);
        }
        pipeline.fusion().setBodyModel(model);

        size_t bones = 0;
        for (size_t i = 0; i < kJointCount; ++i) {
            if (static_cast<Joint>(i) == Joint::Hips)
                continue;
            if (model.boneLengthToParent[i] > 0.0f)
                ++bones;
        }
        setProgress(total, total);
        if (!saved) {
            finishJob(false, 0.0,
                      "estimated " + std::to_string(bones) +
                          " bone lengths and applied them, but saving failed: " + calibPath);
            return;
        }
        finishJob(true, 0.0,
                  "estimated " + std::to_string(bones) + " bone lengths from " +
                      std::to_string(frames) + " fused frames; saved and applied");
    }

    void runPlayspaceJob(const std::string& hand, double seconds) {
        ProgressFn progress = [this](size_t done, size_t needed, const std::string& phase) {
            std::lock_guard<std::mutex> lk(jobMutex);
            job.done = done;
            job.needed = needed;
            if (!phase.empty())
                job.phase = phase;
        };

        std::string err;
        const RigidFit fit = opt.playspace(hand, seconds, progress, jobCancel, err);

        if (!fit.ok) {
            if (jobCancel.load())
                finishJob(false, 0.0, "cancelled");
            else
                finishJob(false, 0.0,
                          err.empty() ? std::string("playspace calibration failed") : err);
            return;
        }

        bool saved = false;
        {
            std::lock_guard<std::mutex> lk(storeMutex);
            store.setWorldAnchor(fit.transform);
            saved = store.save(calibPath);
        }
        const double rmseCm = static_cast<double>(fit.rmse) * 100.0;
        if (!saved) {
            finishJob(false, rmseCm, "anchor solved but saving failed: " + calibPath);
            return;
        }
        finishJob(true, rmseCm,
                  "world anchor saved (rmse " + fmt2(rmseCm) +
                      " cm over " + std::to_string(fit.samples) +
                      " samples); the SteamVR bridge applies it on the next pipeline start");
    }

    // ---------------------------------------------------------- POST routes

    void handleCalibratePair(const httplib::Request& req, httplib::Response& res) {
        json body;
        std::string perr;
        if (!parseBody(req, body, perr)) {
            sendError(res, 400, perr);
            return;
        }
        const std::string refId = jsonString(body, "reference");
        const std::string tgtId = jsonString(body, "target");
        if (refId.empty() || tgtId.empty()) {
            sendError(res, 400, "reference and target node ids are required");
            return;
        }
        if (refId == tgtId) {
            sendError(res, 400, "reference and target must be different nodes");
            return;
        }
        if (!hasNode(refId) || !hasNode(tgtId)) {
            sendError(res, 400,
                      "unknown node id '" + (hasNode(refId) ? tgtId : refId) + "'");
            return;
        }
        double minSamplesD = 200.0;
        double maxSeconds = 60.0;
        if (!jsonNumber(body, "min_samples", minSamplesD) || minSamplesD < 1.0) {
            sendError(res, 400, "min_samples must be a positive number");
            return;
        }
        if (!jsonNumber(body, "max_seconds", maxSeconds) || maxSeconds <= 0.0) {
            sendError(res, 400, "max_seconds must be a positive number");
            return;
        }
        const size_t minSamples = static_cast<size_t>(minSamplesD);

        const bool started = beginJob("pair", "collecting", minSamples,
                                      [this, refId, tgtId, minSamples, maxSeconds] {
                                          runPairJob(refId, tgtId, minSamples, maxSeconds);
                                      });
        if (!started) {
            sendError(res, 200, "busy");
            return;
        }
        sendOk(res);
    }

    void handleCalibrateBody(const httplib::Request& req, httplib::Response& res) {
        json body;
        std::string perr;
        if (!parseBody(req, body, perr)) {
            sendError(res, 400, perr);
            return;
        }
        double seconds = 15.0;
        if (!jsonNumber(body, "seconds", seconds) || seconds <= 0.0) {
            sendError(res, 400, "seconds must be a positive number");
            return;
        }
        const size_t total = std::max<size_t>(static_cast<size_t>(seconds + 0.5), 1);
        const bool started =
            beginJob("body", "collecting", total, [this, seconds] { runBodyJob(seconds); });
        if (!started) {
            sendError(res, 200, "busy");
            return;
        }
        sendOk(res);
    }

    void handleCalibratePlayspace(const httplib::Request& req, httplib::Response& res) {
        if (!opt.playspace) {
            sendError(res, 501, "openvr client not built");
            return;
        }
        json body;
        std::string perr;
        if (!parseBody(req, body, perr)) {
            sendError(res, 400, perr);
            return;
        }
        std::string hand = jsonString(body, "hand");
        if (hand.empty())
            hand = "right";
        if (hand != "left" && hand != "right") {
            sendError(res, 400, "hand must be 'left' or 'right'");
            return;
        }
        double seconds = 20.0;
        if (!jsonNumber(body, "seconds", seconds) || seconds <= 0.0) {
            sendError(res, 400, "seconds must be a positive number");
            return;
        }
        const bool started = beginJob("playspace", "starting", 0,
                                      [this, hand, seconds] { runPlayspaceJob(hand, seconds); });
        if (!started) {
            sendError(res, 200, "busy");
            return;
        }
        sendOk(res);
    }

    // --------------------------------------------------------------- routes

    void setupRoutes() {
        svr.set_exception_handler(
            [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
                std::string msg = "internal error";
                try {
                    if (ep)
                        std::rethrow_exception(ep);
                } catch (const std::exception& e) {
                    msg = e.what();
                } catch (...) {
                }
                sendError(res, 500, msg);
            });

        // Unknown routes (and any error a handler left bodiless) get the JSON
        // shape from DASHBOARD_API.md; bodies set by handlers pass through.
        svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
            if (!res.body.empty())
                return;
            res.set_header("Cache-Control", "no-store");
            const std::string msg =
                (res.status == 404) ? "not found" : ("error " + std::to_string(res.status));
            res.set_content(json{{"ok", false}, {"error", msg}}.dump(), "application/json");
        });

        svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
            if (!opt.webDirOverride.empty()) {
                // Dev loop: re-read on every request so edits show on reload.
                std::string html;
                if (readFile(opt.webDirOverride + "/index.html", html)) {
                    res.set_content(html, "text/html");
                    return;
                }
            }
            res.set_content(reinterpret_cast<const char*>(mn::web::kIndexHtml),
                            mn::web::kIndexHtml_len, "text/html");
        });

        svr.Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            sendJson(res, statusJson());
        });

        svr.Get("/api/skeleton", [this](const httplib::Request&, httplib::Response& res) {
            sendJson(res, skeletonJson());
        });

        svr.Get("/api/events", [this](const httplib::Request& req, httplib::Response& res) {
            sendJson(res, eventsJson(req));
        });

        svr.Get("/api/config", [this](const httplib::Request&, httplib::Response& res) {
            sendJson(res, pipeline.appConfig().toJson());
        });

        svr.Get("/api/calibrate/status", [this](const httplib::Request&, httplib::Response& res) {
            sendJson(res, calibrateStatusJson());
        });

        svr.Get("/api/doctor", [this](const httplib::Request&, httplib::Response& res) {
            sendJson(res, doctorJson());
        });

        svr.Post("/api/calibrate/pair", [this](const httplib::Request& req,
                                               httplib::Response& res) {
            handleCalibratePair(req, res);
        });

        svr.Post("/api/calibrate/body", [this](const httplib::Request& req,
                                               httplib::Response& res) {
            handleCalibrateBody(req, res);
        });

        svr.Post("/api/calibrate/playspace", [this](const httplib::Request& req,
                                                    httplib::Response& res) {
            handleCalibratePlayspace(req, res);
        });

        svr.Post("/api/calibrate/cancel", [this](const httplib::Request&,
                                                 httplib::Response& res) {
            jobCancel.store(true);
            sendOk(res);
        });
    }
};

// ---------------------------------------------------------------------------

DashboardServer::DashboardServer(Pipeline& pipeline, CalibrationStore& store,
                                 std::string calibPath, Options opt)
    : impl_(std::make_unique<Impl>(pipeline, store, std::move(calibPath), std::move(opt))) {}

DashboardServer::~DashboardServer() = default;

bool DashboardServer::start() {
    return impl_->startImpl();
}

void DashboardServer::stop() {
    impl_->stopImpl();
}

bool DashboardServer::isRunning() const {
    return impl_->running.load();
}

std::string DashboardServer::url() const {
    return "http://" + impl_->opt.bind + ":" + std::to_string(impl_->opt.port);
}

std::string DashboardServer::lastError() const {
    return impl_->lastError;
}

} // namespace mn::dash
