#pragma once
#include <sstream>
#include <string>

#include <functional>

namespace mn::log {

// Off is a console-filter level only (nothing is ever written AT Off); the
// secondary sink still receives every message when the console is Off.
enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

void setLevel(Level lvl);
Level level();
void write(Level lvl, const std::string& msg);

// Optional secondary sink. Receives EVERY message (regardless of the console
// level filter) so structured consumers (EventLog) see warnings even when the
// console is quiet. Must be cheap and thread-safe; set empty to remove.
using Sink = std::function<void(Level, const std::string&)>;
void setSink(Sink sink);

namespace detail {
template <typename... Args> std::string fmt(Args&&... args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return os.str();
}
} // namespace detail

template <typename... Args> void debug(Args&&... a) {
    write(Level::Debug, detail::fmt(std::forward<Args>(a)...));
}
template <typename... Args> void info(Args&&... a) {
    write(Level::Info, detail::fmt(std::forward<Args>(a)...));
}
template <typename... Args> void warn(Args&&... a) {
    write(Level::Warn, detail::fmt(std::forward<Args>(a)...));
}
template <typename... Args> void error(Args&&... a) {
    write(Level::Error, detail::fmt(std::forward<Args>(a)...));
}

} // namespace mn::log
