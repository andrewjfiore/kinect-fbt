#pragma once
// Capture node abstraction. A node is one sensor (or simulated sensor): it
// emits SkeletonFrames in its own node-local frame (right-handed, +Y up, +Z
// from sensor toward user, meters). The fusion engine transforms node-local
// frames into the world frame using each node's calibrated extrinsic Pose.

#include "mn/skeleton.hpp"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mn {

struct NodeDescriptor {
    std::string id;   // unique, from config ("front", "back-left", ...)
    std::string type; // registry type ("mock", "replay", "kinect_v2", "kinect_v1")
};

// Called from the node's capture thread. Must be cheap and thread-safe.
using FrameCallback = std::function<void(const NodeDescriptor&, const SkeletonFrame&)>;

class ICaptureNode {
public:
    virtual ~ICaptureNode() = default;
    virtual const NodeDescriptor& descriptor() const = 0;
    // Begin emitting frames via `cb`. Returns false on failure (see lastError()).
    virtual bool start(FrameCallback cb) = 0;
    // Must be idempotent and safe to call on a never-started node. Nodes are
    // not required to support start() after stop(); restart is done by
    // recreating the node through its factory (see the pipeline watchdog).
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual std::string lastError() const { return {}; }
};

// Factory: (id, params from config) -> node, or nullptr with `error` set.
using NodeFactory = std::function<std::unique_ptr<ICaptureNode>(
    const std::string& id, const nlohmann::json& params, std::string& error)>;

class NodeRegistry {
public:
    void add(const std::string& type, NodeFactory f) { factories_[type] = std::move(f); }

    std::unique_ptr<ICaptureNode> create(const std::string& type, const std::string& id,
                                         const nlohmann::json& params, std::string& error) const {
        auto it = factories_.find(type);
        if (it == factories_.end()) {
            error = "unknown capture node type: " + type;
            return nullptr;
        }
        return it->second(id, params, error);
    }

    std::vector<std::string> types() const {
        std::vector<std::string> out;
        for (auto& [k, v] : factories_)
            out.push_back(k);
        return out;
    }

private:
    std::map<std::string, NodeFactory> factories_;
};

} // namespace mn
