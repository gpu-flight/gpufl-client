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

    // Every field of schema 1 must be there with its type, and the record
    // must describe a state the gate can publish (the checks below). This is
    // an operational diagnostic: a valid record is a consistent report, not
    // evidence that capture worked.
    const auto has = [&doc](const char* key, const json::JsonValue::Type type) {
        return doc.contains(key) && doc[key].type() == type;
    };
    using Type = json::JsonValue::Type;
    for (const char* key : {"seq", "deadline_monotonic_ns", "callbacks_seen",
                            "handshake_reentries"}) {
        if (!has(key, Type::Int)) return Invalid("malformed");
    }
    if (!has("wait_config_invalid", Type::Bool)) return Invalid("malformed");
    for (const char* key : {"lifecycle", "owner", "readiness", "reason", "protocol"}) {
        if (!has(key, Type::String)) return Invalid("malformed");
    }
    for (const char* key : {"trigger", "worker", "handshake"}) {
        if (!has(key, Type::Object)) return Invalid("malformed");
        for (const char* field : {"expected", "returned", "ok"}) {
            if (!doc[key].contains(field) || !doc[key][field].is_bool()) {
                return Invalid("malformed");
            }
        }
    }
    if (!has("coverage_notes", Type::Array)) return Invalid("malformed");

    BootstrapStatus status;
    status.seq = doc["seq"].get_int();
    status.lifecycle = doc["lifecycle"].get_string();
    status.owner = doc["owner"].get_string();
    status.readiness = doc["readiness"].get_string();
    status.reason = doc["reason"].get_string();
    status.deadline_monotonic_ns = doc["deadline_monotonic_ns"].get_int();
    status.protocol = doc["protocol"].get_string();
    for (const json::JsonValue& note : doc["coverage_notes"].get_array()) {
        if (!note.is_string()) return Invalid("malformed");
        status.coverage_notes.insert(note.get_string());
    }

    const std::set<std::string> lifecycles = {"not_requested", "driver_initializing",
                                              "driver_ready", "capture_initializing",
                                              "finished"};
    const std::set<std::string> owners = {"none", "gate", "callback", "manual"};
    const std::set<std::string> readiness = {"pending", "ready", "failed"};
    if (status.seq < 1 || !lifecycles.count(status.lifecycle) ||
        !owners.count(status.owner) || !readiness.count(status.readiness)) {
        return Invalid("inconsistent");
    }
    const bool requested = status.lifecycle != "not_requested";
    if (requested != (status.owner != "none") ||
        requested != (status.deadline_monotonic_ns > 0) ||
        (status.lifecycle == "finished" && status.readiness == "pending") ||
        (status.readiness == "ready" && status.lifecycle != "finished") ||
        (status.readiness == "failed") == status.reason.empty()) {
        return Invalid("inconsistent");
    }

    // Participants: returned needs expected, ok needs returned; the owner
    // decides which may be expected; Finished needs every expected one
    // returned, and Ready needs every expected one ok.
    struct Participant { bool expected, returned, ok; };
    const auto participant = [&doc](const char* key) {
        const json::JsonValue& p = doc[key];
        return Participant{p["expected"].get_bool(), p["returned"].get_bool(),
                           p["ok"].get_bool()};
    };
    const Participant trigger = participant("trigger");
    const Participant worker = participant("worker");
    const Participant handshake = participant("handshake");
    bool any_expected = false;
    for (const Participant& p : {trigger, worker, handshake}) {
        if ((p.returned && !p.expected) || (p.ok && !p.returned)) {
            return Invalid("inconsistent");
        }
        any_expected = any_expected || p.expected;
        if (status.lifecycle == "finished" && p.expected && !p.returned) {
            return Invalid("inconsistent");
        }
        if (status.readiness == "ready" && p.expected && !p.ok) {
            return Invalid("inconsistent");
        }
    }
    const bool owner_fits =
        status.owner == "none"     ? !any_expected
        : status.owner == "gate"   ? trigger.expected && !handshake.expected
        : status.owner == "callback" ? worker.expected && !trigger.expected &&
                                         !handshake.expected
                                   : handshake.expected && !trigger.expected &&
                                         !worker.expected;
    if (!owner_fits) return Invalid("inconsistent");
    status.valid = true;
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

FinalBootstrapObservation CombineFinalBootstrap(const InitWatchdog& watchdog,
                                                const BootstrapStatus& final_read) {
    FinalBootstrapObservation out;
    if (!watchdog.seen_valid()) {
        out.accepted = final_read;  // invalid; its error says why
        out.final_error = final_read.error;
        return out;
    }
    out.accepted = watchdog.last();
    if (!final_read.valid) {
        out.final_error = final_read.error;
    } else if (final_read.seq < watchdog.last().seq) {
        out.final_error = "stale_final_seq";
    }
    return out;
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
