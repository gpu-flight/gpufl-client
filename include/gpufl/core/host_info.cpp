// Hostname / IP resolution for session telemetry labels.
//
// Encapsulates the per-platform `gethostname()` plumbing in a single
// TU so callers don't have to deal with winsock2 vs <unistd.h>
// differences. We also keep <windows.h> out of this file — pulling it
// in would re-create the winsock1 vs winsock2 ordering footgun we
// already fixed for the HTTP path. winsock2 functions are accessed
// directly via <winsock2.h>.

#include "gpufl/core/host_info.hpp"

#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>          // inet_pton / inet_ntop
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#include <arpa/inet.h>         // inet_addr / inet_ntop
#include <netinet/in.h>        // sockaddr_in
#include <sys/socket.h>        // socket / connect / getsockname
#endif

namespace gpufl {

std::string getLocalHostname() {
    // The cached lambda runs once per process. WSAStartup is
    // reference-counted, so even if cpp-httplib's Client constructor
    // also called it, calling it here is safe (and pairs with the
    // implicit WSACleanup at process exit).
    static const std::string cached = []() -> std::string {
        char buf[256] = {};
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        if (gethostname(buf, sizeof(buf)) == 0) {
            return std::string(buf);
        }
#else
        if (gethostname(buf, sizeof(buf) - 1) == 0) {
            return std::string(buf);
        }
#endif
        return {};
    }();
    return cached;
}

std::string getLocalIpAddr() {
    // The "UDP-connect" trick for resolving the local outbound IP:
    //
    //   1. Create a UDP socket.
    //   2. `connect()` it to any routable external address (we use
    //      8.8.8.8:53 — Google DNS, always reachable in routing tables).
    //      For UDP, `connect()` does NOT send a packet; it just records
    //      the destination so the kernel can pick a source address via
    //      its routing table.
    //   3. `getsockname()` then returns the local IP the kernel would
    //      use for traffic to that destination — i.e. the machine's
    //      primary outbound IP.
    //
    // This is preferable to iterating `GetAdaptersAddresses` /
    // `getifaddrs` and picking "the first non-loopback interface":
    // the routing-table answer is what an outbound HTTP request would
    // actually use, even on hosts with multiple interfaces (LAN + VPN
    // + WSL bridge + Hyper-V virtual switches, etc.).
    //
    // IPv4 only for now — most dashboards display v4. If the host is
    // pure-IPv6 the call returns empty (acceptable starting point).
    //
    // Cached: routing tables can change at runtime (VPN connect /
    // disconnect), but session-level labels don't need to follow
    // that — capture once and stick with it for the session.
    static const std::string cached = []() -> std::string {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock == INVALID_SOCKET) return {};
#else
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) return {};
#endif

        sockaddr_in target;
        std::memset(&target, 0, sizeof(target));
        target.sin_family = AF_INET;
        target.sin_addr.s_addr = inet_addr("8.8.8.8");
        target.sin_port = htons(53);

        std::string out;
        if (connect(sock,
                    reinterpret_cast<sockaddr*>(&target),
                    sizeof(target)) == 0) {
            sockaddr_in self;
            std::memset(&self, 0, sizeof(self));
#ifdef _WIN32
            int selfLen = sizeof(self);
#else
            socklen_t selfLen = sizeof(self);
#endif
            if (getsockname(sock,
                            reinterpret_cast<sockaddr*>(&self),
                            &selfLen) == 0) {
                char buf[INET_ADDRSTRLEN] = {};
                if (inet_ntop(AF_INET, &self.sin_addr,
                              buf, sizeof(buf)) != nullptr) {
                    out = buf;
                    // 0.0.0.0 means "kernel didn't pick" (no route);
                    // 127.x is loopback (uninformative). Treat both
                    // as "no usable IP" so the JSON stays clean.
                    if (out.rfind("127.", 0) == 0 ||
                        out == "0.0.0.0") {
                        out.clear();
                    }
                }
            }
        }

#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return out;
    }();
    return cached;
}

}  // namespace gpufl
