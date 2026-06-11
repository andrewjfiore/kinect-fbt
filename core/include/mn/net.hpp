#pragma once
// Minimal cross-platform UDP socket (Winsock / POSIX). Enough for the OSC
// endpoint, the OpenVR driver bridge, and tests.

#include <cstddef>
#include <cstdint>
#include <string>

namespace mn {

class UdpSocket {
public:
    UdpSocket() = default;
    ~UdpSocket();
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    UdpSocket(UdpSocket&& other) noexcept;
    UdpSocket& operator=(UdpSocket&& other) noexcept;

    // Resolve host (IPv4 dotted or hostname) and set the default send target.
    bool openSend(const std::string& host, uint16_t port);
    // Bind for receiving. bindAddr "127.0.0.1" (default) or "0.0.0.0".
    bool openReceive(uint16_t port, const std::string& bindAddr = "127.0.0.1");

    // Returns bytes sent or -1.
    int send(const void* data, size_t len);
    // Returns bytes received, 0 on timeout, -1 on error. timeoutMs < 0 blocks.
    int receive(void* buf, size_t cap, int timeoutMs);

    bool isOpen() const { return fd_ != kInvalid; }
    void close();
    std::string lastError() const { return lastError_; }

    // Idempotent global init (WSAStartup on Windows; no-op elsewhere).
    static bool globalInit();

private:
    static constexpr intptr_t kInvalid = -1;
    intptr_t fd_ = kInvalid;
    std::string lastError_;
    // default destination for send()
    uint32_t destAddrBE_ = 0; // network byte order
    uint16_t destPortBE_ = 0; // network byte order
};

} // namespace mn
