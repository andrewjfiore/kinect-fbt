#include "mn/net.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
#define MN_INVALID_SOCK INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
#define MN_INVALID_SOCK (-1)
#endif

#include <cstring>
#include <mutex>

namespace mn {

namespace {
std::string sysError() {
#ifdef _WIN32
    return "winsock error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

bool resolveIPv4(const std::string& host, uint32_t& addrBE, std::string& err) {
    in_addr direct{};
    if (inet_pton(AF_INET, host.c_str(), &direct) == 1) {
        addrBE = direct.s_addr;
        return true;
    }
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
        err = "cannot resolve host: " + host;
        return false;
    }
    addrBE = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr.s_addr;
    freeaddrinfo(res);
    return true;
}
} // namespace

bool UdpSocket::globalInit() {
#ifdef _WIN32
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        WSADATA wsa{};
        ok = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    });
    return ok;
#else
    return true;
#endif
}

UdpSocket::~UdpSocket() { close(); }

UdpSocket::UdpSocket(UdpSocket&& other) noexcept { *this = std::move(other); }

UdpSocket& UdpSocket::operator=(UdpSocket&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        destAddrBE_ = other.destAddrBE_;
        destPortBE_ = other.destPortBE_;
        lastError_ = std::move(other.lastError_);
        other.fd_ = kInvalid;
    }
    return *this;
}

bool UdpSocket::openSend(const std::string& host, uint16_t port) {
    if (!globalInit()) {
        lastError_ = "socket library init failed";
        return false;
    }
    close();
    if (!resolveIPv4(host, destAddrBE_, lastError_))
        return false;
    destPortBE_ = htons(port);
    const sock_t s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == MN_INVALID_SOCK) {
        lastError_ = sysError();
        return false;
    }
    fd_ = static_cast<intptr_t>(s);
    return true;
}

bool UdpSocket::openReceive(uint16_t port, const std::string& bindAddr) {
    if (!globalInit()) {
        lastError_ = "socket library init failed";
        return false;
    }
    close();
    const sock_t s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == MN_INVALID_SOCK) {
        lastError_ = sysError();
        return false;
    }
    int reuse = 1;
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse),
                 sizeof(reuse));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, bindAddr.c_str(), &addr.sin_addr) != 1) {
        lastError_ = "bad bind address: " + bindAddr;
#ifdef _WIN32
        ::closesocket(s);
#else
        ::close(s);
#endif
        return false;
    }
    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        lastError_ = sysError();
#ifdef _WIN32
        ::closesocket(s);
#else
        ::close(s);
#endif
        return false;
    }
    fd_ = static_cast<intptr_t>(s);
    return true;
}

int UdpSocket::send(const void* data, size_t len) {
    if (fd_ == kInvalid)
        return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = destPortBE_;
    addr.sin_addr.s_addr = destAddrBE_;
    const int n = static_cast<int>(::sendto(static_cast<sock_t>(fd_),
                                            static_cast<const char*>(data),
#ifdef _WIN32
                                            static_cast<int>(len),
#else
                                            len,
#endif
                                            0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)));
    if (n < 0)
        lastError_ = sysError();
    return n;
}

int UdpSocket::receive(void* buf, size_t cap, int timeoutMs) {
    if (fd_ == kInvalid)
        return -1;
    const sock_t s = static_cast<sock_t>(fd_);
    if (timeoutMs >= 0) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(s, &rfds);
        timeval tv{};
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        const int sel = ::select(static_cast<int>(s) + 1, &rfds, nullptr, nullptr, &tv);
        if (sel == 0)
            return 0; // timeout
        if (sel < 0) {
            lastError_ = sysError();
            return -1;
        }
    }
    const int n = static_cast<int>(::recvfrom(s, static_cast<char*>(buf),
#ifdef _WIN32
                                              static_cast<int>(cap),
#else
                                              cap,
#endif
                                              0, nullptr, nullptr));
    if (n < 0)
        lastError_ = sysError();
    return n;
}

void UdpSocket::close() {
    if (fd_ == kInvalid)
        return;
#ifdef _WIN32
    ::closesocket(static_cast<sock_t>(fd_));
#else
    ::close(static_cast<int>(fd_));
#endif
    fd_ = kInvalid;
}

} // namespace mn
