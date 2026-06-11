#pragma once
// Service endpoints consume mapped tracker poses (world frame) and deliver
// them somewhere: VRChat OSC, the OpenVR driver bridge, a recorder, ...

#include "mn/mapping.hpp"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mn {

class IServiceEndpoint {
public:
    virtual ~IServiceEndpoint() = default;
    virtual std::string name() const = 0;
    virtual bool start() = 0; // returns false on failure (see lastError())
    virtual void stop() = 0;
    // Called from the pipeline tick thread at the configured rate. `trackers`
    // may include a TrackerRole::Head entry (used by OSC for VRChat alignment;
    // the OpenVR bridge ignores it).
    virtual void push(const std::vector<TrackerPose>& trackers, double timestamp) = 0;
    virtual std::string lastError() const { return {}; }
};

using EndpointFactory = std::function<std::unique_ptr<IServiceEndpoint>(
    const nlohmann::json& params, std::string& error)>;

class EndpointRegistry {
public:
    void add(const std::string& type, EndpointFactory f) { factories_[type] = std::move(f); }

    std::unique_ptr<IServiceEndpoint> create(const std::string& type,
                                             const nlohmann::json& params,
                                             std::string& error) const {
        auto it = factories_.find(type);
        if (it == factories_.end()) {
            error = "unknown endpoint type: " + type;
            return nullptr;
        }
        return it->second(params, error);
    }

    std::vector<std::string> types() const {
        std::vector<std::string> out;
        for (auto& [k, v] : factories_)
            out.push_back(k);
        return out;
    }

private:
    std::map<std::string, EndpointFactory> factories_;
};

} // namespace mn
