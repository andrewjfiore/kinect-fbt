#include "mn_mock/mock.hpp"

#include "mn/clock.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <utility>

namespace mn::mock {
namespace {

using nlohmann::json;

constexpr float kPi = 3.14159265358979323846f;

// Body proportions (meters). Fixed for all t so every bone length is constant
// in every pattern; fusion tests rely on that. Top of the head sits at ~1.70.
constexpr float kHipsY = 0.95f;
constexpr float kSpineY = 1.10f;
constexpr float kChestY = 1.30f;
constexpr float kNeckY = 1.45f;
constexpr float kHeadY = 1.60f;
constexpr float kShoulderX = 0.18f;
constexpr float kShoulderY = 1.40f;
constexpr float kHipX = 0.09f;
constexpr float kHipJointY = 0.90f;
constexpr float kUpperArm = 0.28f;
constexpr float kForeArm = 0.25f;
constexpr float kThigh = 0.42f;
constexpr float kShank = 0.40f;

// Pattern tuning.
constexpr float kWalkHz = 1.4f;
constexpr float kLegLiftRad = 0.7f;   // max thigh raise during walk_in_place
constexpr float kArmSwingRad = 0.5f;  // arm swing amplitude during walk_in_place
constexpr float kSwayHz = 0.5f;
constexpr float kSwayAmpM = 0.15f;

std::chrono::steady_clock::duration toDuration(double seconds) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(seconds));
}

// ---------------------------------------------------------------------------
// JSON param helpers (lenient: missing key keeps the default, wrong type errors;
// unknown keys are tolerated since configs may carry e.g. a default "extrinsic").

bool jsonNumber(const json& p, const char* key, double& out, std::string& error,
                const char* who) {
    const auto it = p.find(key);
    if (it == p.end())
        return true;
    if (!it->is_number()) {
        error = std::string(who) + ": param '" + key + "' must be a number";
        return false;
    }
    out = it->get<double>();
    return true;
}

bool jsonBool(const json& p, const char* key, bool& out, std::string& error, const char* who) {
    const auto it = p.find(key);
    if (it == p.end())
        return true;
    if (!it->is_boolean()) {
        error = std::string(who) + ": param '" + key + "' must be a boolean";
        return false;
    }
    out = it->get<bool>();
    return true;
}

bool jsonString(const json& p, const char* key, std::string& out, std::string& error,
                const char* who) {
    const auto it = p.find(key);
    if (it == p.end())
        return true;
    if (!it->is_string()) {
        error = std::string(who) + ": param '" + key + "' must be a string";
        return false;
    }
    out = it->get<std::string>();
    return true;
}

bool jsonVec3(const json& a, Vec3& out) {
    if (!a.is_array() || a.size() != 3)
        return false;
    for (const auto& v : a) {
        if (!v.is_number())
            return false;
    }
    out = Vec3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
    return true;
}

// view_pose: { "pos": [x,y,z], "rot": [w,x,y,z] }, both fields optional.
bool parseViewPose(const json& j, Pose& out, std::string& error) {
    if (!j.is_object()) {
        error = "mock: 'view_pose' must be an object {pos:[x,y,z], rot:[w,x,y,z]}";
        return false;
    }
    if (const auto it = j.find("pos"); it != j.end()) {
        if (!jsonVec3(*it, out.pos)) {
            error = "mock: 'view_pose.pos' must be [x,y,z]";
            return false;
        }
    }
    if (const auto it = j.find("rot"); it != j.end()) {
        const json& a = *it;
        if (!a.is_array() || a.size() != 4) {
            error = "mock: 'view_pose.rot' must be [w,x,y,z]";
            return false;
        }
        for (const auto& v : a) {
            if (!v.is_number()) {
                error = "mock: 'view_pose.rot' must be [w,x,y,z]";
                return false;
            }
        }
        const Quat q(a[0].get<float>(), a[1].get<float>(), a[2].get<float>(), a[3].get<float>());
        if (q.coeffs().norm() < 1e-6f) {
            error = "mock: 'view_pose.rot' is degenerate (near-zero quaternion)";
            return false;
        }
        out.rot = q.normalized();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Occlusion simulation: same spirit as the fusion occlusion weight. Estimate
// the torso plane from the frame itself; joints on the far side of that plane
// from the sensor (origin in node-local) become Inferred at reduced confidence.

void applyOcclusionSim(SkeletonFrame& f, float baseConfidence) {
    const Vec3 chest = f[Joint::Chest].pos;
    const Vec3 hips = f[Joint::Hips].pos;
    const Vec3 center = (chest + hips) * 0.5f;
    const Vec3 up = safeNormalized(chest - hips);
    const Vec3 right = safeNormalized(f[Joint::ShoulderR].pos - f[Joint::ShoulderL].pos,
                                      Vec3::UnitX());
    const Vec3 facing = safeNormalized(up.cross(right), -Vec3::UnitZ());
    const float sensorSide = (-center).dot(facing); // sensor sits at the node-local origin
    constexpr float kMargin = 0.04f;
    for (JointSample& s : f.joints) {
        const float side = (s.pos - center).dot(facing);
        if (side * sensorSide < 0.0f && std::abs(side) > kMargin) {
            s.state = TrackState::Inferred;
            s.confidence = baseConfidence * 0.4f;
        }
    }
}

// ---------------------------------------------------------------------------
// Mock node.

struct MockParams {
    std::string pattern = "walk_in_place";
    double rateHz = 30.0;
    double noiseM = 0.0;
    double seed = 1.0;
    double confidence = 0.9;
    bool occlusionSim = false;
    Pose viewPose{};
};

class MockNode final : public ICaptureNode {
public:
    MockNode(std::string id, MockParams params)
        : desc_{std::move(id), "mock"}, params_(std::move(params)) {}

    ~MockNode() override { stop(); }

    const NodeDescriptor& descriptor() const override { return desc_; }

    bool start(FrameCallback cb) override {
        if (running_) {
            lastError_ = "mock: node already running";
            return false;
        }
        if (!cb) {
            lastError_ = "mock: null frame callback";
            return false;
        }
        if (thread_.joinable())
            thread_.join(); // reap a previously finished thread
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopRequested_ = false;
        }
        cb_ = std::move(cb);
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }

    void stop() override {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopRequested_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable())
            thread_.join();
        running_ = false;
    }

    bool isRunning() const override { return running_.load(); }
    std::string lastError() const override { return lastError_; }

private:
    void run() {
        const Pose invView = params_.viewPose.inverse();
        const float noise = static_cast<float>(params_.noiseM);
        const float conf = static_cast<float>(params_.confidence);
        std::mt19937 rng(static_cast<uint32_t>(static_cast<long long>(params_.seed)));
        std::normal_distribution<float> gauss(0.0f, noise > 0.0f ? noise : 1.0f);
        const auto period = toDuration(1.0 / params_.rateHz);
        const double t0 = nowSeconds();
        auto next = std::chrono::steady_clock::now();
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (cv_.wait_until(lk, next, [this] { return stopRequested_; }))
                    break;
            }
            SkeletonFrame f = groundTruth(nowSeconds() - t0, params_.pattern);
            for (JointSample& s : f.joints) {
                s.pos = invView.apply(s.pos); // world -> node-local, positions only
                if (noise > 0.0f)
                    s.pos += Vec3(gauss(rng), gauss(rng), gauss(rng));
                s.confidence = conf;
            }
            if (params_.occlusionSim)
                applyOcclusionSim(f, conf);
            f.timestamp = nowSeconds();
            cb_(desc_, f);
            next += period;
            const auto now = std::chrono::steady_clock::now();
            if (next < now)
                next = now; // do not burst to catch up after a stall
        }
        running_ = false;
    }

    NodeDescriptor desc_;
    MockParams params_;
    FrameCallback cb_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stopRequested_ = false;
    std::string lastError_;
};

std::unique_ptr<ICaptureNode> makeMockNode(const std::string& id, const json& params,
                                           std::string& error) {
    const json p = params.is_null() ? json::object() : params;
    if (!p.is_object()) {
        error = "mock: params must be a JSON object";
        return nullptr;
    }
    MockParams mp;
    if (!jsonString(p, "pattern", mp.pattern, error, "mock"))
        return nullptr;
    if (mp.pattern != "walk_in_place" && mp.pattern != "tpose" && mp.pattern != "sway") {
        error = "mock: unknown pattern '" + mp.pattern +
                "' (expected walk_in_place | tpose | sway)";
        return nullptr;
    }
    if (!jsonNumber(p, "rate_hz", mp.rateHz, error, "mock"))
        return nullptr;
    if (!std::isfinite(mp.rateHz) || mp.rateHz <= 0.0 || mp.rateHz > 10000.0) {
        error = "mock: 'rate_hz' must be a positive frame rate";
        return nullptr;
    }
    if (!jsonNumber(p, "noise_m", mp.noiseM, error, "mock"))
        return nullptr;
    if (!std::isfinite(mp.noiseM) || mp.noiseM < 0.0) {
        error = "mock: 'noise_m' must be >= 0";
        return nullptr;
    }
    if (!jsonNumber(p, "seed", mp.seed, error, "mock"))
        return nullptr;
    if (!std::isfinite(mp.seed)) {
        error = "mock: 'seed' must be a finite number";
        return nullptr;
    }
    if (!jsonNumber(p, "confidence", mp.confidence, error, "mock"))
        return nullptr;
    if (!std::isfinite(mp.confidence) || mp.confidence < 0.0 || mp.confidence > 1.0) {
        error = "mock: 'confidence' must be within [0,1]";
        return nullptr;
    }
    if (!jsonBool(p, "occlusion_sim", mp.occlusionSim, error, "mock"))
        return nullptr;
    if (const auto it = p.find("view_pose"); it != p.end()) {
        if (!parseViewPose(*it, mp.viewPose, error))
            return nullptr;
    }
    return std::make_unique<MockNode>(id, std::move(mp));
}

// ---------------------------------------------------------------------------
// Replay node.

class ReplayNode final : public ICaptureNode {
public:
    ReplayNode(std::string id, std::vector<SkeletonFrame> frames, bool loop, double speed)
        : desc_{std::move(id), "replay"}, frames_(std::move(frames)), loop_(loop), speed_(speed) {
        if (frames_.size() > 1) {
            const double span = frames_.back().timestamp - frames_.front().timestamp;
            if (span > 0.0)
                wrapDelta_ = span / static_cast<double>(frames_.size() - 1);
        }
    }

    ~ReplayNode() override { stop(); }

    const NodeDescriptor& descriptor() const override { return desc_; }

    bool start(FrameCallback cb) override {
        if (running_) {
            lastError_ = "replay: node already running";
            return false;
        }
        if (!cb) {
            lastError_ = "replay: null frame callback";
            return false;
        }
        if (thread_.joinable())
            thread_.join(); // reap a previously finished thread (e.g. loop=false ended)
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopRequested_ = false;
        }
        cb_ = std::move(cb);
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }

    void stop() override {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopRequested_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable())
            thread_.join();
        running_ = false;
    }

    bool isRunning() const override { return running_.load(); }
    std::string lastError() const override { return lastError_; }

private:
    void run() {
        size_t i = 0;
        auto next = std::chrono::steady_clock::now();
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (cv_.wait_until(lk, next, [this] { return stopRequested_; }))
                    break;
            }
            SkeletonFrame f = frames_[i];
            f.timestamp = nowSeconds(); // restamp; stored t only drives pacing
            cb_(desc_, f);
            size_t ni = i + 1;
            double dt = 0.0;
            if (ni >= frames_.size()) {
                if (!loop_)
                    break; // end of file, no looping: thread ends by itself
                ni = 0;
                dt = wrapDelta_;
            } else {
                dt = frames_[ni].timestamp - frames_[i].timestamp;
            }
            dt = (dt > 0.0 ? dt : 0.0) / speed_;
            next += toDuration(dt);
            const auto now = std::chrono::steady_clock::now();
            if (next < now)
                next = now;
            i = ni;
        }
        running_ = false;
    }

    NodeDescriptor desc_;
    std::vector<SkeletonFrame> frames_;
    bool loop_ = true;
    double speed_ = 1.0;
    double wrapDelta_ = 1.0 / 30.0;
    FrameCallback cb_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stopRequested_ = false;
    std::string lastError_;
};

std::unique_ptr<ICaptureNode> makeReplayNode(const std::string& id, const json& params,
                                             std::string& error) {
    const json p = params.is_null() ? json::object() : params;
    if (!p.is_object()) {
        error = "replay: params must be a JSON object";
        return nullptr;
    }
    std::string file;
    if (!jsonString(p, "file", file, error, "replay"))
        return nullptr;
    if (file.empty()) {
        error = "replay: param 'file' is required";
        return nullptr;
    }
    bool loop = true;
    if (!jsonBool(p, "loop", loop, error, "replay"))
        return nullptr;
    double speed = 1.0;
    if (!jsonNumber(p, "speed", speed, error, "replay"))
        return nullptr;
    if (!std::isfinite(speed) || speed <= 0.0) {
        error = "replay: 'speed' must be > 0";
        return nullptr;
    }
    std::vector<SkeletonFrame> frames;
    if (!loadJsonl(file, frames)) {
        error = "replay: cannot load JSONL file '" + file + "'";
        return nullptr;
    }
    if (frames.empty()) {
        error = "replay: file '" + file + "' contains no frames";
        return nullptr;
    }
    return std::make_unique<ReplayNode>(id, std::move(frames), loop, speed);
}

} // namespace

// ---------------------------------------------------------------------------
// Ground truth.

SkeletonFrame groundTruth(double t, const std::string& pattern) {
    SkeletonFrame f;
    f.timestamp = t;
    f.hasBody = true;

    const auto set = [&f](Joint j, const Vec3& p) {
        JointSample& s = f[j];
        s.pos = p;
        s.rot = Quat::Identity();
        s.confidence = 1.0f;
        s.state = TrackState::Tracked;
        s.hasRot = false;
    };

    // The body stands at the origin facing -Z with +Y up (world convention).
    // Its left = up x forward = (+Y) x (-Z) = -X, so left joints sit at x < 0.
    set(Joint::Hips, Vec3(0.0f, kHipsY, 0.0f));
    set(Joint::Spine, Vec3(0.0f, kSpineY, 0.0f));
    set(Joint::Chest, Vec3(0.0f, kChestY, 0.0f));
    set(Joint::Neck, Vec3(0.0f, kNeckY, 0.0f));
    set(Joint::Head, Vec3(0.0f, kHeadY, 0.0f));

    const Vec3 shoulderL(-kShoulderX, kShoulderY, 0.0f);
    const Vec3 shoulderR(kShoulderX, kShoulderY, 0.0f);
    const Vec3 hipL(-kHipX, kHipJointY, 0.0f);
    const Vec3 hipR(kHipX, kHipJointY, 0.0f);
    const Vec3 footOffset(0.0f, -0.03f, -0.15f); // ankle -> toe, toes point forward (-Z)

    set(Joint::ShoulderL, shoulderL);
    set(Joint::ShoulderR, shoulderR);
    set(Joint::HipL, hipL);
    set(Joint::HipR, hipR);

    // Leg: thigh pivots about the hip (around +X) by `lift`; the shank stays
    // vertical and the foot offset is rigid, so every bone length is constant
    // and knee/ankle/foot rise and fall together.
    const auto leg = [&](Joint knee, Joint ankle, Joint foot, const Vec3& hip, float lift) {
        const Vec3 kneePos = hip + Vec3(0.0f, -kThigh * std::cos(lift), -kThigh * std::sin(lift));
        const Vec3 anklePos = kneePos + Vec3(0.0f, -kShank, 0.0f);
        set(knee, kneePos);
        set(ankle, anklePos);
        set(foot, anklePos + footOffset);
    };

    // Arm: straight arm swinging about the shoulder (around +X); positive
    // `swing` moves the wrist forward (-Z). swing = 0 hangs straight down.
    const auto armSwing = [&](Joint elbow, Joint wrist, const Vec3& shoulder, float swing) {
        const Vec3 dir(0.0f, -std::cos(swing), -std::sin(swing));
        const Vec3 elbowPos = shoulder + kUpperArm * dir;
        set(elbow, elbowPos);
        set(wrist, elbowPos + kForeArm * dir);
    };

    if (pattern == "tpose") {
        const auto armT = [&](Joint elbow, Joint wrist, const Vec3& shoulder, float side) {
            const Vec3 elbowPos = shoulder + Vec3(side * kUpperArm, 0.0f, 0.0f);
            set(elbow, elbowPos);
            set(wrist, elbowPos + Vec3(side * kForeArm, 0.0f, 0.0f));
        };
        armT(Joint::ElbowL, Joint::WristL, shoulderL, -1.0f);
        armT(Joint::ElbowR, Joint::WristR, shoulderR, 1.0f);
        leg(Joint::KneeL, Joint::AnkleL, Joint::FootL, hipL, 0.0f);
        leg(Joint::KneeR, Joint::AnkleR, Joint::FootR, hipR, 0.0f);
    } else if (pattern == "sway") {
        armSwing(Joint::ElbowL, Joint::WristL, shoulderL, 0.0f);
        armSwing(Joint::ElbowR, Joint::WristR, shoulderR, 0.0f);
        leg(Joint::KneeL, Joint::AnkleL, Joint::FootL, hipL, 0.0f);
        leg(Joint::KneeR, Joint::AnkleR, Joint::FootR, hipR, 0.0f);
        const float dx = kSwayAmpM * std::sin(2.0f * kPi * kSwayHz * static_cast<float>(t));
        for (JointSample& s : f.joints)
            s.pos.x() += dx; // hips translate laterally; the whole body follows
    } else {
        // walk_in_place (also the fallback for unknown patterns; the mock
        // factory validates the pattern name before we get here).
        const float phase = 2.0f * kPi * kWalkHz * static_cast<float>(t);
        const float osc = std::sin(phase);
        const float liftL = kLegLiftRad * 0.5f * (1.0f + osc);
        const float liftR = kLegLiftRad * 0.5f * (1.0f - osc);
        armSwing(Joint::ElbowL, Joint::WristL, shoulderL, -kArmSwingRad * osc);
        armSwing(Joint::ElbowR, Joint::WristR, shoulderR, kArmSwingRad * osc);
        leg(Joint::KneeL, Joint::AnkleL, Joint::FootL, hipL, liftL);
        leg(Joint::KneeR, Joint::AnkleR, Joint::FootR, hipR, liftR);
    }
    return f;
}

// ---------------------------------------------------------------------------
// JSONL recording / loading. One frame per line:
//   {"t": <double>, "has_body": true,
//    "joints": [[x,y,z, qw,qx,qy,qz, confidence, state], ... kJointCount entries]}

JsonlRecorder::~JsonlRecorder() { close(); }

bool JsonlRecorder::open(const std::string& path) {
    close();
    count_ = 0;
#ifdef _WIN32
    if (fopen_s(&f_, path.c_str(), "wb") != 0)
        f_ = nullptr;
#else
    f_ = std::fopen(path.c_str(), "wb");
#endif
    return f_ != nullptr;
}

void JsonlRecorder::write(const SkeletonFrame& frame) {
    if (!f_)
        return;
    json joints = json::array();
    for (const JointSample& s : frame.joints) {
        joints.push_back({s.pos.x(), s.pos.y(), s.pos.z(), s.rot.w(), s.rot.x(), s.rot.y(),
                          s.rot.z(), s.confidence, static_cast<int>(s.state)});
    }
    const json line{{"t", frame.timestamp}, {"has_body", frame.hasBody},
                    {"joints", std::move(joints)}};
    std::string text = line.dump();
    text.push_back('\n');
    std::fputs(text.c_str(), f_);
    ++count_;
}

void JsonlRecorder::close() {
    if (f_) {
        std::fclose(f_);
        f_ = nullptr;
    }
}

bool loadJsonl(const std::string& path, std::vector<SkeletonFrame>& out) {
    out.clear();
    std::ifstream in(path);
    if (!in.is_open())
        return false;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty())
            continue;
        const json j = json::parse(line, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object())
            return false;
        const auto tIt = j.find("t");
        const auto bodyIt = j.find("has_body");
        const auto jointsIt = j.find("joints");
        if (tIt == j.end() || !tIt->is_number() || bodyIt == j.end() || !bodyIt->is_boolean() ||
            jointsIt == j.end() || !jointsIt->is_array() || jointsIt->size() != kJointCount)
            return false;
        SkeletonFrame f;
        f.timestamp = tIt->get<double>();
        f.hasBody = bodyIt->get<bool>();
        for (size_t i = 0; i < kJointCount; ++i) {
            const json& e = (*jointsIt)[i];
            if (!e.is_array() || e.size() != 9)
                return false;
            for (const auto& v : e) {
                if (!v.is_number())
                    return false;
            }
            JointSample& s = f.joints[i];
            s.pos = Vec3(e[0].get<float>(), e[1].get<float>(), e[2].get<float>());
            s.rot = Quat(e[3].get<float>(), e[4].get<float>(), e[5].get<float>(),
                         e[6].get<float>());
            s.confidence = e[7].get<float>();
            const int st = e[8].get<int>();
            s.state = (st >= 0 && st <= 2) ? static_cast<TrackState>(st)
                                           : TrackState::NotTracked;
            s.hasRot = false; // the line format does not carry hasRot
        }
        out.push_back(f);
    }
    return true;
}

// ---------------------------------------------------------------------------

void registerNodes(NodeRegistry& reg) {
    reg.add("mock", &makeMockNode);
    reg.add("replay", &makeReplayNode);
}

} // namespace mn::mock
