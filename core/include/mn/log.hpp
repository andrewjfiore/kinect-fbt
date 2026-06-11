#pragma once
#include <sstream>
#include <string>

namespace mn::log {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void setLevel(Level lvl);
Level level();
void write(Level lvl, const std::string& msg);

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
