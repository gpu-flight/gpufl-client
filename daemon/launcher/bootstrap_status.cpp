#include "bootstrap_status.hpp"

#include <chrono>
#include <fstream>
#include <limits>
#include <random>

#include "gpufl/core/json/json.hpp"

#if !defined(_WIN32)
#include <time.h>
#endif

namespace gpufl::launcher {
namespace {

constexpr std::size_t kMaxStatusBytes = 4096;

BootstrapStatus Invalid(std::string why) {
    BootstrapStatus status;
    status.error = std::move(why);
    return status;
}

}  // namespace

std::filesystem::path BootstrapStatusPath(const std::filesystem::path& dir,
                                          const long pid) {
    return dir / ("status-" + std::to_string(pid) + ".json");
}

BootstrapStatus ReadBootstrapStatus(const std::filesystem::path& dir,
                                    const long pid, const std::string& nonce) {
    std::ifstream in(BootstrapStatusPath(dir, pid), std::ios::binary);
    if (!in) return Invalid("missing");
    std::string text(kMaxStatusBytes + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (text.size() > kMaxStatusBytes) return Invalid("oversized");

    const json::JsonValue doc = json::parseJson(text);
    if (!doc.is_object()) return Invalid("malformed");
    if (doc.value<int64_t>("schema", 0) != 1) return Invalid("schema");
    if (doc.value<std::string>("nonce", "") != nonce || nonce.empty()) {
        return Invalid("foreign_nonce");
    }
    if (doc.value<int64_t>("pid", 0) != pid) return Invalid("foreign_pid");

    BootstrapStatus status;
    status.valid = true;
    status.seq = doc.value<int64_t>("seq", 0);
    status.lifecycle = doc.value<std::string>("lifecycle", "");
    status.owner = doc.value<std::string>("owner", "");
    status.readiness = doc.value<std::string>("readiness", "");
    status.reason = doc.value<std::string>("reason", "");
    status.deadline_monotonic_ns = doc.value<int64_t>("deadline_monotonic_ns", 0);
    status.protocol = doc.value<std::string>("protocol", "");
    if (doc.contains("coverage_notes") && doc["coverage_notes"].is_array()) {
        for (const json::JsonValue& note : doc["coverage_notes"].get_array()) {
            if (note.is_string()) status.coverage_notes.insert(note.get_string());
        }
    }
    if (status.lifecycle.empty() || status.readiness.empty()) {
        return Invalid("malformed");
    }
    return status;
}

int64_t MonotonicNowNs() {
#if defined(_WIN32)
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#else
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
#endif
}

bool InitWatchdog::Observe(const BootstrapStatus& status, const int64_t now_ns) {
    if (status.valid) {
        if (status.seq >= last_seq_) {
            last_seq_ = status.seq;
            last_ = status;
            seen_valid_ = true;
            if (status.lifecycle == "finished") disarmed_ = true;
            if (!armed_ && status.lifecycle != "not_requested" &&
                status.deadline_monotonic_ns > 0) {
                armed_ = true;
                deadline_ns_ = status.deadline_monotonic_ns;
            }
        }
    }
    if (!armed_ || disarmed_) return false;
    const int64_t max = std::numeric_limits<int64_t>::max();
    const int64_t limit = deadline_ns_ > max - grace_ns_ ? max : deadline_ns_ + grace_ns_;
    return now_ns > limit;
}

std::string DescribeBootstrapStatus(const BootstrapStatus& status) {
    if (!status.valid) return "unavailable (" + status.error + ")";
    std::string out = status.readiness + " owner=" + status.owner +
                      " lifecycle=" + status.lifecycle;
    if (!status.reason.empty()) out += " reason=" + status.reason;
    if (!status.coverage_notes.empty()) {
        out += " coverage=";
        bool first = true;
        for (const std::string& note : status.coverage_notes) {
            if (!first) out += ',';
            first = false;
            out += note;
        }
    }
    return out;
}

std::string MakeBootstrapNonce() {
    std::random_device device;
    static const char* kHex = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 4; ++i) {
        const uint32_t word = device();
        for (int shift = 28; shift >= 0; shift -= 4) out += kHex[(word >> shift) & 0xf];
    }
    return out;
}

}  // namespace gpufl::launcher
