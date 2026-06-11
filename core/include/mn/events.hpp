#pragma once
// Structured event/error tracking: a global, thread-safe ring buffer of
// recent events, fed by the logging layer (and directly by subsystems).
// Consumed by the dashboard (/api/events) and `marionette doctor`.

#include "mn/log.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mn {

struct Event {
    uint64_t seq = 0; // monotonically increasing, starts at 1
    double t = 0.0;   // mn::nowSeconds() at push
    log::Level level = log::Level::Info;
    std::string message;
};

class EventLog {
public:
    static constexpr size_t kCapacity = 2000;

    static EventLog& instance();

    void push(log::Level lvl, const std::string& message);

    // Events with seq > afterSeq, oldest first, capped at maxCount.
    std::vector<Event> since(uint64_t afterSeq, size_t maxCount = 500) const;

    uint64_t latestSeq() const;

    // Lifetime totals by level (Debug, Info, Warn, Error) - not capped by the
    // ring capacity.
    std::array<uint64_t, 4> levelCounts() const;

    void clear(); // drops buffered events and resets counts (seq keeps rising)

    // Route every log::write() into the instance via log::setSink. Idempotent;
    // composes nothing (replaces any previously installed sink).
    static void installLogCapture();

private:
    EventLog() = default;
    struct Impl;
};

} // namespace mn
