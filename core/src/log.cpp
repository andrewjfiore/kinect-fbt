#include "mn/log.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace mn::log {

namespace {
std::atomic<Level> g_level{Level::Info};
std::mutex g_mutex;

const char* tag(Level l) {
    switch (l) {
    case Level::Debug: return "DBG";
    case Level::Info: return "INF";
    case Level::Warn: return "WRN";
    case Level::Error: return "ERR";
    }
    return "???";
}
} // namespace

void setLevel(Level lvl) { g_level.store(lvl); }
Level level() { return g_level.load(); }

void write(Level lvl, const std::string& msg) {
    if (lvl < g_level.load())
        return;
    std::lock_guard<std::mutex> lk(g_mutex);
    std::fprintf(lvl >= Level::Warn ? stderr : stdout, "[%s] %s\n", tag(lvl), msg.c_str());
    std::fflush(lvl >= Level::Warn ? stderr : stdout);
}

} // namespace mn::log
