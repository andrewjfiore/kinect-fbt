#pragma once
// App configuration (config.json) and the calibration store
// (calibration.json: node extrinsics + body model + playspace anchor).

#include "mn/fusion.hpp"
#include "mn/mapping.hpp"

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mn {

// JSON forms: Vec3 = [x,y,z]; Quat = [w,x,y,z];
// Pose = {"pos":[x,y,z], "rot":[w,x,y,z]} (rot optional, defaults identity).
void to_json(nlohmann::json& j, const Pose& p);
void from_json(const nlohmann::json& j, Pose& p);

struct NodeConfigEntry {
    std::string id;
    std::string type;
    nlohmann::json params; // backend-specific; may carry "extrinsic" default
};

struct EndpointConfigEntry {
    std::string type; // "osc", "openvr"
    nlohmann::json params;
};

struct AppConfig {
    std::vector<NodeConfigEntry> nodes;
    std::vector<EndpointConfigEntry> endpoints;
    FusionConfig fusion;
    MappingConfig mapping;
    double tickHz = 90.0;
    std::string calibrationFile = "calibration.json";

    // Throws std::runtime_error with a readable message on bad config.
    static AppConfig load(const std::string& path);
    static AppConfig fromJson(const nlohmann::json& j);
    nlohmann::json toJson() const;
};

class CalibrationStore {
public:
    // Missing file -> empty store (ok=true). Malformed file -> ok=false.
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    std::optional<Pose> nodeExtrinsic(const std::string& nodeId) const;
    void setNodeExtrinsic(const std::string& nodeId, const Pose& p);

    const BodyModel& bodyModel() const { return bodyModel_; }
    void setBodyModel(const BodyModel& m) { bodyModel_ = m; }

    // Marionette world -> SteamVR playspace (identity until calibrated).
    std::optional<Pose> worldAnchor() const { return worldAnchor_; }
    void setWorldAnchor(const Pose& p) { worldAnchor_ = p; }

private:
    std::map<std::string, Pose> extrinsics_;
    BodyModel bodyModel_;
    std::optional<Pose> worldAnchor_;
};

} // namespace mn
