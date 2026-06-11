#include "mn/config.hpp"

#include "mn/log.hpp"
#include "mn/skeleton.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mn {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& where, const std::string& what) {
    throw std::runtime_error(where + ": " + what);
}

double getNumber(const json& j, const char* key, double def, const std::string& ctx) {
    if (!j.contains(key))
        return def;
    const json& v = j.at(key);
    if (!v.is_number())
        fail(ctx + "." + key, "must be a number");
    return v.get<double>();
}

float getFloat(const json& j, const char* key, float def, const std::string& ctx) {
    return static_cast<float>(getNumber(j, key, static_cast<double>(def), ctx));
}

bool getBool(const json& j, const char* key, bool def, const std::string& ctx) {
    if (!j.contains(key))
        return def;
    const json& v = j.at(key);
    if (!v.is_boolean())
        fail(ctx + "." + key, "must be a boolean");
    return v.get<bool>();
}

std::string getString(const json& j, const char* key, std::string def, const std::string& ctx) {
    if (!j.contains(key))
        return def;
    const json& v = j.at(key);
    if (!v.is_string())
        fail(ctx + "." + key, "must be a string");
    return v.get<std::string>();
}

std::string requireString(const json& j, const char* key, const std::string& ctx) {
    if (!j.contains(key))
        fail(ctx, std::string("missing required field \"") + key + "\"");
    const json& v = j.at(key);
    if (!v.is_string() || v.get<std::string>().empty())
        fail(ctx + "." + key, "must be a non-empty string");
    return v.get<std::string>();
}

} // namespace

// ---------------------------------------------------------------------------
// Pose <-> JSON

void to_json(nlohmann::json& j, const Pose& p) {
    j = nlohmann::json{{"pos", {p.pos.x(), p.pos.y(), p.pos.z()}},
                       {"rot", {p.rot.w(), p.rot.x(), p.rot.y(), p.rot.z()}}};
}

void from_json(const nlohmann::json& j, Pose& p) {
    if (!j.is_object())
        fail("Pose", "must be an object {\"pos\":[x,y,z],\"rot\":[w,x,y,z]}");
    if (!j.contains("pos"))
        fail("Pose", "missing required field \"pos\"");
    const json& pos = j.at("pos");
    if (!pos.is_array() || pos.size() != 3 || !pos[0].is_number() || !pos[1].is_number() ||
        !pos[2].is_number())
        fail("Pose.pos", "must be [x,y,z]");
    p.pos = Vec3(pos[0].get<float>(), pos[1].get<float>(), pos[2].get<float>());
    p.rot = Quat::Identity();
    if (j.contains("rot")) {
        const json& rot = j.at("rot");
        if (!rot.is_array() || rot.size() != 4 || !rot[0].is_number() || !rot[1].is_number() ||
            !rot[2].is_number() || !rot[3].is_number())
            fail("Pose.rot", "must be [w,x,y,z]");
        const Quat q(rot[0].get<float>(), rot[1].get<float>(), rot[2].get<float>(),
                     rot[3].get<float>());
        if (q.norm() < 1e-6f)
            fail("Pose.rot", "quaternion is (near) zero");
        p.rot = q.normalized();
    }
}

// ---------------------------------------------------------------------------
// AppConfig

AppConfig AppConfig::fromJson(const nlohmann::json& j) {
    const std::string ctx = "config";
    AppConfig cfg;
    if (!j.is_object())
        fail(ctx, "root must be a JSON object");

    cfg.tickHz = getNumber(j, "tick_hz", cfg.tickHz, ctx);
    if (!(cfg.tickHz > 0.0))
        fail(ctx + ".tick_hz", "must be > 0");
    cfg.calibrationFile = getString(j, "calibration_file", cfg.calibrationFile, ctx);

    if (j.contains("nodes")) {
        const json& arr = j.at("nodes");
        if (!arr.is_array())
            fail(ctx + ".nodes", "must be an array");
        for (size_t i = 0; i < arr.size(); ++i) {
            const std::string nctx = ctx + ".nodes[" + std::to_string(i) + "]";
            const json& n = arr[i];
            if (!n.is_object())
                fail(nctx, "must be an object");
            NodeConfigEntry e;
            e.id = requireString(n, "id", nctx);
            e.type = requireString(n, "type", nctx);
            e.params = json::object();
            if (n.contains("params")) {
                if (!n.at("params").is_object())
                    fail(nctx + ".params", "must be an object");
                e.params = n.at("params");
            }
            cfg.nodes.push_back(std::move(e));
        }
    }

    if (j.contains("endpoints")) {
        const json& arr = j.at("endpoints");
        if (!arr.is_array())
            fail(ctx + ".endpoints", "must be an array");
        for (size_t i = 0; i < arr.size(); ++i) {
            const std::string ectx = ctx + ".endpoints[" + std::to_string(i) + "]";
            const json& n = arr[i];
            if (!n.is_object())
                fail(ectx, "must be an object");
            EndpointConfigEntry e;
            e.type = requireString(n, "type", ectx);
            e.params = json::object();
            if (n.contains("params")) {
                if (!n.at("params").is_object())
                    fail(ectx + ".params", "must be an object");
                e.params = n.at("params");
            }
            cfg.endpoints.push_back(std::move(e));
        }
    }

    if (j.contains("fusion")) {
        const json& f = j.at("fusion");
        if (!f.is_object())
            fail(ctx + ".fusion", "must be an object");
        const std::string fctx = ctx + ".fusion";
        cfg.fusion.staleSeconds = getNumber(f, "stale_seconds", cfg.fusion.staleSeconds, fctx);
        cfg.fusion.inferredWeight = getFloat(f, "inferred_weight", cfg.fusion.inferredWeight, fctx);
        cfg.fusion.minConfidence = getFloat(f, "min_confidence", cfg.fusion.minConfidence, fctx);
        cfg.fusion.depthNoiseRefMeters =
            getFloat(f, "depth_noise_ref_m", cfg.fusion.depthNoiseRefMeters, fctx);
        cfg.fusion.occlusionPenalty =
            getFloat(f, "occlusion_penalty", cfg.fusion.occlusionPenalty, fctx);
        cfg.fusion.boneLengthConstraint =
            getBool(f, "bone_length_constraint", cfg.fusion.boneLengthConstraint, fctx);
        if (f.contains("filter")) {
            const json& fl = f.at("filter");
            if (!fl.is_object())
                fail(fctx + ".filter", "must be an object");
            const std::string flctx = fctx + ".filter";
            cfg.fusion.filter.minCutoff =
                getFloat(fl, "min_cutoff", cfg.fusion.filter.minCutoff, flctx);
            cfg.fusion.filter.beta = getFloat(fl, "beta", cfg.fusion.filter.beta, flctx);
            cfg.fusion.filter.dCutoff = getFloat(fl, "d_cutoff", cfg.fusion.filter.dCutoff, flctx);
        }
    }

    if (j.contains("mapping")) {
        const json& m = j.at("mapping");
        if (!m.is_object())
            fail(ctx + ".mapping", "must be an object");
        const std::string mctx = ctx + ".mapping";
        if (m.contains("trackers")) {
            const json& arr = m.at("trackers");
            if (!arr.is_array())
                fail(mctx + ".trackers", "must be an array of role names");
            std::vector<TrackerRole> roles;
            roles.reserve(arr.size());
            for (size_t i = 0; i < arr.size(); ++i) {
                const std::string tctx = mctx + ".trackers[" + std::to_string(i) + "]";
                if (!arr[i].is_string())
                    fail(tctx, "must be a string");
                const std::string name = arr[i].get<std::string>();
                const auto role = trackerRoleFromName(name);
                if (!role)
                    fail(tctx, "unknown tracker role \"" + name + "\"");
                roles.push_back(*role);
            }
            cfg.mapping.roles = std::move(roles);
        }
        cfg.mapping.emitHead = getBool(m, "emit_head", cfg.mapping.emitHead, mctx);
        cfg.mapping.velocitySmooth =
            getFloat(m, "velocity_smooth", cfg.mapping.velocitySmooth, mctx);
    }

    return cfg;
}

AppConfig AppConfig::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("config: cannot open \"" + path + "\"");
    json j;
    try {
        // Allow // and /* */ comments so documented jsonc examples load as-is.
        j = json::parse(in, nullptr, true, true);
    } catch (const json::parse_error& e) {
        throw std::runtime_error("config: \"" + path + "\": " + e.what());
    }
    try {
        return fromJson(j);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error("\"" + path + "\": " + e.what());
    }
}

nlohmann::json AppConfig::toJson() const {
    json nodesArr = json::array();
    for (const auto& n : nodes)
        nodesArr.push_back(json{{"id", n.id},
                                {"type", n.type},
                                {"params", n.params.is_null() ? json::object() : n.params}});

    json epArr = json::array();
    for (const auto& e : endpoints)
        epArr.push_back(
            json{{"type", e.type}, {"params", e.params.is_null() ? json::object() : e.params}});

    json trackers = json::array();
    for (const TrackerRole r : mapping.roles)
        trackers.push_back(trackerRoleName(r));

    return json{{"tick_hz", tickHz},
                {"calibration_file", calibrationFile},
                {"nodes", nodesArr},
                {"endpoints", epArr},
                {"fusion",
                 {{"stale_seconds", fusion.staleSeconds},
                  {"inferred_weight", fusion.inferredWeight},
                  {"min_confidence", fusion.minConfidence},
                  {"depth_noise_ref_m", fusion.depthNoiseRefMeters},
                  {"occlusion_penalty", fusion.occlusionPenalty},
                  {"bone_length_constraint", fusion.boneLengthConstraint},
                  {"filter",
                   {{"min_cutoff", fusion.filter.minCutoff},
                    {"beta", fusion.filter.beta},
                    {"d_cutoff", fusion.filter.dCutoff}}}}},
                {"mapping",
                 {{"trackers", trackers},
                  {"emit_head", mapping.emitHead},
                  {"velocity_smooth", mapping.velocitySmooth}}}};
}

// ---------------------------------------------------------------------------
// CalibrationStore

bool CalibrationStore::load(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(path), ec)) {
        // Missing file: a fresh, empty (but valid) store.
        extrinsics_.clear();
        bodyModel_ = BodyModel{};
        worldAnchor_.reset();
        return true;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    const json j = json::parse(in, nullptr, false, true);
    if (j.is_discarded() || !j.is_object())
        return false;

    try {
        std::map<std::string, Pose> ext;
        BodyModel bm{};
        std::optional<Pose> anchor;

        if (j.contains("extrinsics")) {
            const json& e = j.at("extrinsics");
            if (!e.is_object())
                return false;
            for (const auto& [id, val] : e.items())
                ext[id] = val.get<Pose>();
        }

        if (j.contains("body_model")) {
            const json& b = j.at("body_model");
            if (!b.is_object())
                return false;
            if (b.contains("valid") && !b.at("valid").is_boolean())
                return false;
            bm.valid = b.value("valid", false);
            if (b.contains("bone_lengths")) {
                const json& bl = b.at("bone_lengths");
                if (!bl.is_object())
                    return false;
                for (const auto& [name, val] : bl.items()) {
                    const auto joint = jointFromName(name);
                    if (!joint) {
                        // Forward compatibility: skip joints this build does not know.
                        log::warn("calibration: ignoring unknown joint \"", name, "\"");
                        continue;
                    }
                    if (!val.is_number())
                        return false;
                    bm.boneLengthToParent[static_cast<size_t>(*joint)] = val.get<float>();
                }
            }
        }

        if (j.contains("world_anchor"))
            anchor = j.at("world_anchor").get<Pose>();

        // Commit only after the whole file parsed cleanly.
        extrinsics_ = std::move(ext);
        bodyModel_ = bm;
        worldAnchor_ = anchor;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool CalibrationStore::save(const std::string& path) const {
    json ext = json::object();
    for (const auto& [id, p] : extrinsics_)
        ext[id] = p;

    json bl = json::object();
    for (size_t i = 0; i < kJointCount; ++i) {
        if (static_cast<Joint>(i) == Joint::Hips)
            continue; // root: boneLengthToParent unused
        const float len = bodyModel_.boneLengthToParent[i];
        if (len > 0.0f)
            bl[jointName(static_cast<Joint>(i))] = len;
    }

    json j{{"extrinsics", ext},
           {"body_model", {{"valid", bodyModel_.valid}, {"bone_lengths", bl}}}};
    if (worldAnchor_)
        j["world_anchor"] = *worldAnchor_;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out << j.dump(2) << '\n';
    return static_cast<bool>(out);
}

std::optional<Pose> CalibrationStore::nodeExtrinsic(const std::string& nodeId) const {
    const auto it = extrinsics_.find(nodeId);
    if (it == extrinsics_.end())
        return std::nullopt;
    return it->second;
}

void CalibrationStore::setNodeExtrinsic(const std::string& nodeId, const Pose& p) {
    extrinsics_[nodeId] = p;
}

} // namespace mn
