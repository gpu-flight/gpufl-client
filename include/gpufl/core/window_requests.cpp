#include "gpufl/core/window_requests.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <system_error>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl::detail {

namespace {

constexpr const char* kRequestSchema = "gpufl.window_request/1";
constexpr const char* kResultSchema = "gpufl.window_result/1";
constexpr int64_t kHourNs = 3'600'000'000'000;
/// Bigger than any real request; a larger file is not one.
constexpr std::uintmax_t kMaxRequestBytes = 64 * 1024;

std::mutex g_mu;
std::unique_ptr<WindowRequestChannel> g_channel;

/// Also the file name, so nothing outside these characters reaches a path.
bool SafeRequestId(const std::string& id) {
    static const std::regex kId("[A-Za-z0-9_-]{1,64}");
    return std::regex_match(id, kId);
}

const char* EnvOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0] != '\0') ? v : nullptr;
}

/// Integer from env, or the fallback when unset; a malformed value is reported, never replaced.
int64_t EnvIntOr(const char* name, const int64_t fallback, std::string* bad) {
    const char* v = EnvOrNull(name);
    if (!v) return fallback;
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || parsed < 0) {
        if (bad->empty()) *bad = std::string(name) + "='" + v + "'";
        return fallback;
    }
    return parsed;
}

std::string StatusFor(const OpenRequestStatus status) {
    switch (status) {
        case OpenRequestStatus::Busy:              return "busy";
        case OpenRequestStatus::Cooldown:          return "cooldown";
        case OpenRequestStatus::EngineUnavailable: return "engine_unavailable";
        default:                                   return "invalid";
    }
}

}  // namespace

WindowRequestChannel::Hooks WindowRequestChannel::liveHooks() {
    Hooks hooks;
    hooks.request_open = [](void*, const DeepWindowSpec& spec) {
        return DeepWindow::RequestOpenTagged(spec);
    };
    hooks.last_opened_token = [](void*) { return DeepWindow::LastOpenedToken(); };
    hooks.pending_open_token = [](void*) { return DeepWindow::PendingOpenToken(); };
    return hooks;
}

std::filesystem::path WindowRequestChannel::DirectoryFor(const std::string& log_dir,
                                                         const std::string& launch_id) {
    return std::filesystem::path(log_dir) / ".gpufl-window-requests" / launch_id;
}

WindowRequestChannel::WindowRequestChannel(std::filesystem::path dir, const int64_t pid,
                                           const WindowRequestLimits limits, const Hooks hooks)
    : dir_(std::move(dir)), pid_(pid), limits_(limits), hooks_(hooks) {}

void WindowRequestChannel::poll(const int64_t now_ns) {
    settleQueued(now_ns);
    std::error_code ec;
    if (!std::filesystem::is_directory(dir_, ec)) return;
    for (std::filesystem::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        // A writer publishes through a dot-named temporary and a rename, so a
        // dot file is never a finished request, and results are this
        // channel's own answers.
        if (name.empty() || name[0] == '.') continue;
        if (name.size() <= 5 || name.compare(name.size() - 5, 5, ".json") != 0) continue;
        if (name.find(".result.json") != std::string::npos) continue;
        const std::string request_id = name.substr(0, name.size() - 5);
        if (!SafeRequestId(request_id)) continue;
        if (answered_.count(request_id) || queued_.count(request_id)) continue;
        consider(it->path(), request_id, now_ns);
    }
}

void WindowRequestChannel::settleQueued(const int64_t now_ns) {
    for (auto it = queued_.begin(); it != queued_.end();) {
        const Queued q = it->second;
        if (hooks_.last_opened_token(hooks_.ctx) == q.token) {
            opened_.emplace_back(now_ns, q.window_ms);
            writeResult(it->first, "opened", q.token, q.window_ms, now_ns);
        } else if (hooks_.pending_open_token(hooks_.ctx) == q.token) {
            ++it;
            continue;
        } else {
            // Neither queued nor the last to open: something else took the
            // coordinator first. Reported, so the asker learns it got nothing.
            writeResult(it->first, "busy", 0, q.window_ms, now_ns);
        }
        answered_.insert(it->first);
        it = queued_.erase(it);
    }
}

void WindowRequestChannel::consider(const std::filesystem::path& file,
                                    const std::string& request_id, const int64_t now_ns) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(file, ec);
    json::JsonValue request;
    if (!ec && size <= kMaxRequestBytes) request = json::loadFile(file.string());
    if (!request.is_object() || request.value<std::string>("schema", "") != kRequestSchema ||
        request.value<std::string>("request_id", "") != request_id) {
        answered_.insert(request_id);
        writeResult(request_id, "invalid", 0, 0, now_ns);
        return;
    }
    const int64_t asked_ms = request.value<int64_t>("duration_ms", 0);
    const int64_t requested_ns = request.value<int64_t>("requested_ns", 0);
    const int64_t expires_ns = request.value<int64_t>("expires_ns", 0);
    if (asked_ms <= 0 || expires_ns <= requested_ns) {
        answered_.insert(request_id);
        writeResult(request_id, "invalid", 0, 0, now_ns);
        return;
    }
    if (now_ns > expires_ns) {
        answered_.insert(request_id);
        writeResult(request_id, "expired", 0, 0, now_ns);
        return;
    }
    const int64_t window_ms = std::min(asked_ms, limits_.max_window_ms);
    if (!withinBudget(window_ms, now_ns)) {
        answered_.insert(request_id);
        writeResult(request_id, "budget", 0, window_ms, now_ns);
        return;
    }

    DeepWindowSpec spec;
    spec.max_duration_ms = window_ms;
    spec.cooldown_ms = limits_.cooldown_ms;
    spec.name = "window_request";
    spec.trigger.present = true;
    spec.trigger.source = "request";
    spec.trigger.request_id = request_id;
    spec.trigger.rule_id = request.value<std::string>("rule_id", "");
    spec.trigger.expression = request.value<std::string>("expression", "");
    spec.trigger.fired_ns = requested_ns;

    const OpenRequestResult result = hooks_.request_open(hooks_.ctx, spec);
    if (result.status == OpenRequestStatus::PreparationPending) {
        // No CUDA context yet - the API server, or an engine still starting.
        // Not answered: if this process becomes a GPU process before the
        // request expires, it acts then.
        return;
    }
    if (result.status == OpenRequestStatus::Accepted && result.token != 0) {
        queued_[request_id] = Queued{result.token, window_ms};
        return;
    }
    answered_.insert(request_id);
    writeResult(request_id, StatusFor(result.status), 0, window_ms, now_ns);
}

bool WindowRequestChannel::withinBudget(const int64_t window_ms, const int64_t now_ns) {
    while (!opened_.empty() && now_ns - opened_.front().first >= kHourNs) opened_.pop_front();
    int64_t capture_ms = 0;
    for (const auto& [at_ns, ms] : opened_) capture_ms += ms;
    return static_cast<int64_t>(opened_.size()) < limits_.max_windows_per_hour &&
           capture_ms + window_ms <= limits_.max_capture_ms_per_hour;
}

void WindowRequestChannel::writeResult(const std::string& request_id, const std::string& status,
                                       const uint64_t token, const int64_t window_ms,
                                       const int64_t now_ns) const {
    const std::string name = request_id + "." + std::to_string(pid_) + ".result.json";
    const std::filesystem::path final_path = dir_ / name;
    const std::filesystem::path temp_path = dir_ / ("." + name + ".tmp");
    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            GFL_LOG_ERROR("[WindowRequest] cannot write the answer to ", request_id);
            return;
        }
        out << "{\"schema\":\"" << kResultSchema << "\""
            << ",\"request_id\":\"" << json::escape(request_id) << "\""
            << ",\"pid\":" << pid_
            << ",\"status\":\"" << status << "\""
            << ",\"token\":" << token
            << ",\"window_ms\":" << window_ms
            << ",\"decided_ns\":" << now_ns << "}";
    }
    std::error_code ec;
    // Within one directory a rename is atomic, so the asker reads a whole answer or none.
    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        GFL_LOG_ERROR("[WindowRequest] cannot publish the answer to ", request_id, ": ", ec.message());
        std::filesystem::remove(temp_path, ec);
    }
}

void WindowRequestChannel::InstallFromEnv(const std::string& log_dir) {
    std::lock_guard lk(g_mu);
    g_channel.reset();
    if (const char* enabled = EnvOrNull(env::kWindowRequests);
        enabled && std::string(enabled) == "0") {
        return;
    }
    const char* launch_id = EnvOrNull(env::kLaunchId);
    if (!launch_id || log_dir.empty() || !SafeRequestId(launch_id)) return;

    std::string bad;
    WindowRequestLimits limits;
    limits.max_window_ms = EnvIntOr(env::kWindowRequestMaxMs, limits.max_window_ms, &bad);
    limits.max_windows_per_hour =
        EnvIntOr(env::kWindowRequestMaxPerHour, limits.max_windows_per_hour, &bad);
    limits.max_capture_ms_per_hour =
        EnvIntOr(env::kWindowRequestMaxCaptureMsPerHour, limits.max_capture_ms_per_hour, &bad);
    limits.cooldown_ms = EnvIntOr(env::kWindowRequestCooldownMs, limits.cooldown_ms, &bad);
    if (!bad.empty()) {
        GFL_LOG_ERROR("[WindowRequest] ", bad,
                      " is not a non-negative integer; window requests are off for this process");
        return;
    }
    g_channel = std::make_unique<WindowRequestChannel>(
        DirectoryFor(log_dir, launch_id), static_cast<int64_t>(GetPid()), limits, liveHooks());
}

void WindowRequestChannel::Service() {
    std::lock_guard lk(g_mu);
    if (g_channel) g_channel->poll(GetTimestampNs());
}

bool WindowRequestChannel::Installed() {
    std::lock_guard lk(g_mu);
    return g_channel != nullptr;
}

void WindowRequestChannel::ResetForTesting() {
    std::lock_guard lk(g_mu);
    g_channel.reset();
}

}  // namespace gpufl::detail
