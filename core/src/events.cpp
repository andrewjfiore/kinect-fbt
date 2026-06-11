#include "mn/events.hpp"

#include "mn/clock.hpp"

#include <algorithm>
#include <deque>
#include <mutex>

namespace mn {

// All state lives in a single process-wide Impl (Meyers singleton); EventLog
// itself is a stateless facade so instance() stays trivially cheap.
struct EventLog::Impl {
    mutable std::mutex mutex;
    std::deque<Event> ring;                // ordered by seq, capped at kCapacity
    uint64_t nextSeq = 1;                  // never reset; seq keeps rising across clear()
    std::array<uint64_t, 4> counts{};      // lifetime totals by level

    static Impl& get() {
        static Impl impl;
        return impl;
    }
};

EventLog& EventLog::instance() {
    static EventLog inst;
    return inst;
}

void EventLog::push(log::Level lvl, const std::string& message) {
    Impl& im = Impl::get();
    std::lock_guard<std::mutex> lk(im.mutex);
    Event e;
    e.seq = im.nextSeq++;
    e.t = nowSeconds();
    e.level = lvl;
    e.message = message;
    const size_t idx = static_cast<size_t>(lvl);
    if (idx < im.counts.size())
        ++im.counts[idx];
    im.ring.push_back(std::move(e));
    while (im.ring.size() > kCapacity)
        im.ring.pop_front();
}

std::vector<Event> EventLog::since(uint64_t afterSeq, size_t maxCount) const {
    Impl& im = Impl::get();
    std::lock_guard<std::mutex> lk(im.mutex);
    std::vector<Event> out;
    // The ring is ordered by seq: binary-search the first event past afterSeq.
    auto it = std::lower_bound(im.ring.begin(), im.ring.end(), afterSeq,
                               [](const Event& e, uint64_t s) { return e.seq <= s; });
    for (; it != im.ring.end() && out.size() < maxCount; ++it)
        out.push_back(*it);
    return out;
}

uint64_t EventLog::latestSeq() const {
    Impl& im = Impl::get();
    std::lock_guard<std::mutex> lk(im.mutex);
    return im.nextSeq - 1;
}

std::array<uint64_t, 4> EventLog::levelCounts() const {
    Impl& im = Impl::get();
    std::lock_guard<std::mutex> lk(im.mutex);
    return im.counts;
}

void EventLog::clear() {
    Impl& im = Impl::get();
    std::lock_guard<std::mutex> lk(im.mutex);
    im.ring.clear();
    im.counts.fill(0);
    // nextSeq intentionally untouched: seq keeps rising across clear().
}

void EventLog::installLogCapture() {
    // Replaces any previously installed sink; installing twice is harmless
    // (the sink routes into the same singleton either way).
    log::setSink([](log::Level lvl, const std::string& msg) {
        EventLog::instance().push(lvl, msg);
    });
}

} // namespace mn
