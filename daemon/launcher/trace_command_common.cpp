#include "trace_command_common.hpp"

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "agent_launcher.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/log_salvage.hpp"
#include "gpufl/inject/inject_entry.hpp"
#include "gpufl/upload/upload_logs.hpp"

namespace gpufl::launcher {
namespace {

fs::path findInjectLib(const TracePlatform& platform, const fs::path& exe) {
    for (const auto& c : platform.injectLibCandidates(exe)) {
        std::error_code ec;
        if (fs::exists(c, ec)) return fs::weakly_canonical(c, ec);
    }
    return {};
}

bool setEnvOrPrint(const TracePlatform& platform,
                   const char* key,
                   const std::string& value) {
    std::string error;
    if (platform.setEnv(key, value, error)) return true;
    std::fprintf(stderr, "gpufl: %s\n", error.c_str());
    return false;
}

// A '+'-joined pass token ("Trace+PcSampling") runs those engines together in
// one process via GPUFL_ENGINE_COMBO. Returns the comma-joined combo for a
// composite token, or "" for a single-engine token.
std::string comboForToken(const std::string& token) {
    if (token.find('+') == std::string::npos) return {};
    std::string combo = token;
    std::replace(combo.begin(), combo.end(), '+', ',');
    return combo;
}

std::string firstEngineOfToken(const std::string& token) {
    const size_t plus = token.find('+');
    return plus == std::string::npos ? token : token.substr(0, plus);
}

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    uint64_t v = rng();
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (v & 0xffffffffu);
    return os.str();
}

std::string makeAnalysisId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        0x9e3779b97f4a7c15ULL};
    const uint64_t a = rng();
    const uint64_t b = rng();
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
                  static_cast<unsigned>(a >> 32),
                  static_cast<unsigned>((a >> 16) & 0xffff),
                  static_cast<unsigned>(a & 0xffff),
                  static_cast<unsigned>((b >> 48) & 0xffff),
                  static_cast<unsigned long long>(b & 0xffffffffffffULL));
    return buf;
}

// Filesystem-safe slug for a run-folder name: keep [A-Za-z0-9-_.], turn anything else into '-',
// trim leading/trailing dashes, cap length. Empty input falls back to "session".
std::string slugForPath(const std::string& in) {
    auto safe = [](const char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    };
    std::string out;
    for (const char c : in) out.push_back(safe(c) ? c : '-');
    while (!out.empty() && out.front() == '-') out.erase(out.begin());
    while (!out.empty() && out.back() == '-') out.pop_back();
    if (out.size() > 48) out.resize(48);
    return out.empty() ? std::string("session") : out;
}

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

bool isPlainLog(const fs::path& path) {
    return path.extension() == ".log";
}

bool isGzipLog(const fs::path& path) {
    if (path.extension() != ".gz") return false;
    return path.stem().extension() == ".log";
}

template <typename Fn>
void readPlainLines(const fs::path& path, Fn&& fn) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        fn(line);
    }
}

template <typename Fn>
void readGzipLines(const fs::path& path, Fn&& fn) {
    gzFile gz = gzopen(path.string().c_str(), "rb");
    if (!gz) return;

    char buf[4096];
    std::string line;
    while (gzgets(gz, buf, sizeof(buf)) != nullptr) {
        line.append(buf);
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            fn(line);
            line.clear();
        }
    }
    if (!line.empty()) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        fn(line);
    }
    gzclose(gz);
}

template <typename Fn>
void readLogLines(const fs::path& path, Fn&& fn) {
    if (isGzipLog(path)) {
        readGzipLines(path, std::forward<Fn>(fn));
    } else if (isPlainLog(path)) {
        readPlainLines(path, std::forward<Fn>(fn));
    }
}

struct SessionLifecycleInfo {
    bool saw_job_start = false;
    bool saw_shutdown = false;
    int pid = 0;
    std::string app = "unknown";
    int64_t job_start_ts_ns = 0;
};

bool isSessionDirectory(const fs::directory_entry& entry) {
    std::error_code ec;
    if (!entry.is_directory(ec)) return false;
    const std::string name = entry.path().filename().string();
    return !name.empty() && name.front() != '.';
}

struct SyntheticShutdownContext {
    int exit_code = 0;
    bool signaled = false;
    int signal = 0;
    bool window_stopped = false;  // intentional window-end stop, not a crash
};

void inspectLifecycleLine(const std::string& line, SessionLifecycleInfo& info) {
    if (line.find("\"type\":\"shutdown\"") != std::string::npos) {
        info.saw_shutdown = true;
        return;
    }
    if (info.saw_job_start ||
        line.find("\"type\":\"job_start\"") == std::string::npos) {
        return;
    }

    const json::JsonValue doc = json::parseJson(line);
    if (!doc.is_object()) return;

    const std::string type = doc.value<std::string>("type", "");
    if (type != "job_start" || info.saw_job_start) return;

    info.saw_job_start = true;
    info.pid = doc.value<int>("pid", 0);
    info.app = doc.value<std::string>("app", "unknown");
    info.job_start_ts_ns = doc.value<int64_t>("ts_ns", 0);
}

bool decompressGzipToFile(const fs::path& gz_path, const fs::path& out_path) {
    gzFile gz = gzopen(gz_path.string().c_str(), "rb");
    if (!gz) return false;
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        gzclose(gz);
        return false;
    }

    char buf[64 * 1024];
    int n = 0;
    while ((n = gzread(gz, buf, sizeof(buf))) > 0) {
        out.write(buf, n);
    }
    const int err = gzclose(gz);
    out.close();
    return n == 0 && err == Z_OK && out.good();
}

std::string syntheticShutdownLine(const std::string& session_id,
                                  const SessionLifecycleInfo& info,
                                  const SyntheticShutdownContext& context) {
    const bool crashed = !context.window_stopped &&
                         (context.signaled || context.exit_code != 0);
    std::ostringstream os;
    os << "{\"type\":\"shutdown\""
       << ",\"pid\":" << info.pid
       << ",\"app\":\"" << json::escape(info.app) << "\""
       << ",\"session_id\":\"" << json::escape(session_id) << "\""
       << ",\"ts_ns\":" << nowNs()
       << ",\"synthetic\":true"
       << ",\"exit_code\":" << context.exit_code
       << ",\"signaled\":" << (context.signaled ? "true" : "false")
       << ",\"signal\":" << context.signal
       << ",\"window_stopped\":" << (context.window_stopped ? "true" : "false")
       << ",\"crashed\":" << (crashed ? "true" : "false")
       << "}";
    return os.str();
}

bool appendLine(const fs::path& path, const std::string& line) {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.is_open()) return false;
    out << line << '\n';
    return out.good();
}

bool appendLineToGzipLog(const fs::path& gz_path, const std::string& line) {
    const fs::path dir = gz_path.parent_path();
    const fs::path tmp_plain = dir / ".gpufl.synthetic-shutdown.system.log";
    const fs::path tmp_gz(tmp_plain.string() + ".gz");

    std::error_code ec;
    fs::remove(tmp_plain, ec);
    fs::remove(tmp_gz, ec);

    if (!decompressGzipToFile(gz_path, tmp_plain)) {
        fs::remove(tmp_plain, ec);
        return false;
    }
    if (!appendLine(tmp_plain, line)) {
        fs::remove(tmp_plain, ec);
        return false;
    }

    GzipFileCompressor compressor;
    if (!compressor.compress(tmp_plain.string())) {
        fs::remove(tmp_plain, ec);
        fs::remove(tmp_gz, ec);
        return false;
    }
    fs::remove(tmp_plain, ec);

    const fs::path backup(gz_path.string() + ".bak");
    fs::remove(backup, ec);
    fs::rename(gz_path, backup, ec);
    if (ec) {
        fs::remove(tmp_gz, ec);
        return false;
    }
    fs::rename(tmp_gz, gz_path, ec);
    if (ec) {
        std::error_code restore_ec;
        fs::rename(backup, gz_path, restore_ec);
        fs::remove(tmp_gz, ec);
        return false;
    }
    fs::remove(backup, ec);
    fs::remove(tmp_plain, ec);
    return true;
}

// Highest existing system.<k>.log[.gz] window index under session_dir (0 if none).
int highestSystemWindowIndex(const fs::path& session_dir) {
    int max_idx = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(session_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        std::string name = entry.path().filename().string();
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".gz") == 0) {
            name.resize(name.size() - 3);
        }
        if (name.size() <= 4 || name.compare(name.size() - 4, 4, ".log") != 0) continue;
        name.resize(name.size() - 4);                  // "system" or "system.<k>"
        if (name.rfind("system.", 0) != 0) continue;   // skip the non-indexed "system"
        const std::string suffix = name.substr(std::string("system.").size());
        if (suffix.empty() ||
            !std::all_of(suffix.begin(), suffix.end(),
                         [](unsigned char c) { return std::isdigit(c); })) {
            continue;
        }
        try { max_idx = std::max(max_idx, std::stoi(suffix)); } catch (...) {}
    }
    return max_idx;
}

bool appendSyntheticShutdown(const fs::path& session_dir,
                             const std::string& session_id,
                             const SessionLifecycleInfo& info,
                             const SyntheticShutdownContext& context) {
    const std::string line = syntheticShutdownLine(session_id, info, context);
    const std::string base = "system." + std::to_string(highestSystemWindowIndex(session_dir) + 1) + ".log";
    const fs::path window_gz = session_dir / (base + ".gz");
    std::error_code ec;
    if (fs::exists(window_gz, ec)) {
        return appendLineToGzipLog(window_gz, line);
    }
    return appendLine(session_dir / base, line);
}

bool isSystemLog(const fs::path& path) {
    if (!isPlainLog(path) && !isGzipLog(path)) return false;

    std::string name = path.filename().string();
    if (name.size() > 3 && name.compare(name.size() - 3, 3, ".gz") == 0) {
        name.resize(name.size() - 3);
    }
    if (name.size() <= 4 || name.compare(name.size() - 4, 4, ".log") != 0) {
        return false;
    }
    name.resize(name.size() - 4);

    if (name == "system") return true;
    if (name.rfind("system.", 0) != 0) return false;
    const std::string suffix = name.substr(std::string("system.").size());
    if (suffix.empty()) return false;
    return std::all_of(suffix.begin(), suffix.end(),
                       [](unsigned char c) { return std::isdigit(c); });
}

void inspectLifecycleFiles(const std::vector<fs::path>& paths,
                           SessionLifecycleInfo& info) {
    for (const fs::path& path : paths) {
        readLogLines(path, [&](const std::string& line) {
            inspectLifecycleLine(line, info);
        });
        if (info.saw_shutdown) break;
    }
}

int ensureTraceCompletionMarkers(const fs::path& output_dir,
                                 const int64_t min_job_start_ts_ns,
                                 const SyntheticShutdownContext& context) {
    std::error_code ec;
    if (output_dir.empty() || !fs::exists(output_dir, ec)) return 0;

    int synthesized = 0;
    for (const auto& session_entry : fs::directory_iterator(output_dir, ec)) {
        if (ec) break;
        if (!isSessionDirectory(session_entry)) continue;

        const fs::path session_dir = session_entry.path();
        const std::string session_id = session_dir.filename().string();

        std::vector<fs::path> system_logs;
        std::vector<fs::path> fallback_logs;
        SessionLifecycleInfo info;
        std::error_code inner_ec;
        for (const auto& file_entry : fs::directory_iterator(session_dir, inner_ec)) {
            if (inner_ec) break;
            if (!file_entry.is_regular_file(inner_ec)) continue;
            const fs::path path = file_entry.path();
            if (!isPlainLog(path) && !isGzipLog(path)) continue;
            if (isSystemLog(path)) {
                system_logs.push_back(path);
            } else {
                fallback_logs.push_back(path);
            }
        }

        if (!system_logs.empty()) {
            inspectLifecycleFiles(system_logs, info);
        } else {
            inspectLifecycleFiles(fallback_logs, info);
        }

        if (info.saw_job_start && !info.saw_shutdown &&
            info.job_start_ts_ns >= min_job_start_ts_ns &&
            appendSyntheticShutdown(session_dir, session_id, info, context)) {
            ++synthesized;
        }
    }
    return synthesized;
}

// Out-of-process upload-complete signal. After the agent has drained and exited
// cleanly, POST /events/session-complete for each session under output_dir so the
// backend finalizes on its fast path (upload_complete_at) instead of waiting out
// the grace/settle window. The agent sends this too, but from inside a process
// racing its own shutdown — that POST is often interrupted before it lands; this
// launcher-side signal is the reliable one. Best-effort: a failure just means the
// backend's grace path finalizes a little slower.
void signalSessionsComplete(const fs::path& output_dir,
                            const BackendConfig& config,
                            const bool quiet) {
    if (config.backend_url.empty() || config.api_key.empty()) return;

    std::error_code ec;
    if (output_dir.empty() || !fs::exists(output_dir, ec)) return;

    // Fire off requests in parallel. Bound total time via per-POST httplib timeouts
    // (currently ~6s) to ensure a black-holed backend can't stall trace exit.
    std::vector<std::future<std::pair<std::string, bool>>> futures;
    for (const auto& entry : fs::directory_iterator(output_dir, ec)) {
        if (ec) break;
        if (!isSessionDirectory(entry)) continue;

        const std::string session_id = entry.path().filename().string();
        futures.push_back(std::async(std::launch::async, [config, session_id]() {
            return std::make_pair(session_id, postSessionComplete(config, session_id));
        }));
    }

    for (auto& f : futures) {
        const auto res = f.get();
        if (res.second) {
            GFL_LOG_DEBUG("signalled upload-complete for session ", res.first);
        } else if (!quiet) {
            GFL_LOG_ERROR("failed to signal upload-complete for session ", res.first);
        }
    }
}

/// Remove with a short backoff (100/200/400 ms) - a freshly written or
/// freshly released file is often still held briefly on Windows (AV
/// scan, indexer, an uploader that just stopped). True on success OR
/// when the file is already gone.
bool removeWithRetry(const fs::path& p, std::error_code& ec) {
    for (int attempt = 0;; ++attempt) {
        if (fs::remove(p, ec) || !ec) return true;
        if (attempt >= 2) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100 << attempt));
    }
}

int repairUncompressedLogs(const fs::path& root) {
    std::error_code root_ec;
    if (root.empty() || !fs::exists(root, root_ec)) return 0;

    const auto salvaged = salvageSessionTempDirs(root);
    int repaired = salvaged.salvaged;

    GzipFileCompressor compressor;
    std::error_code iter_ec;
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, iter_ec), end;
         !iter_ec && it != end;
         it.increment(iter_ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec)) continue;
        const fs::path path = it->path();
        // .tmp contents were handled by the salvage pass above; a dir
        // that survived it is still held - leave it alone.
        if (path.parent_path().filename() == ".tmp") continue;
        if (path.extension() != ".log") continue;

        std::error_code size_ec;
        const auto size = fs::file_size(path, size_ec);
        if (size_ec) continue;
        if (size == 0) {
            std::error_code remove_ec;
            fs::remove(path, remove_ec);
            continue;
        }

        const fs::path gz_path(path.string() + ".gz");
        std::error_code exists_ec;
        if (fs::exists(gz_path, exists_ec)) {
            std::error_code remove_ec;
            if (!removeWithRetry(path, remove_ec)) {
                // Held without delete sharing - truncate so the data
                // exists only in the .gz; the empty husk is swept by a
                // later pass once the holder lets go.
                std::ofstream trunc(path, std::ios::out | std::ios::trunc);
                if (!trunc) {
                    std::fprintf(stderr,
                                 "[gpufl] warning: could not remove stale %s "
                                 "(%s) - its .gz holds the same data\n",
                                 path.string().c_str(),
                                 remove_ec.message().c_str());
                }
            }
            continue;
        }

        if (compressor.compress(path.string())) {
            // compress() removes (or truncates) the original itself and
            // logs when it can't - nothing left to do here.
            ++repaired;
        }
    }
    return repaired;
}

}  // namespace

int runTraceCommon(const TraceArgs& args, const TracePlatform& platform) {
    const fs::path exe = platform.selfExe();
    if (exe.empty()) {
        std::fprintf(stderr, "gpufl: cannot resolve launcher path (%s)\n",
                     platform.platformName());
        return 3;
    }

    const fs::path inject_lib = findInjectLib(platform, exe);
    if (inject_lib.empty()) {
        std::fprintf(stderr,
                     "gpufl: cannot find %s adjacent to %s\n",
                     platform.injectLibraryName(), exe.string().c_str());
        std::fprintf(stderr, "       searched:\n");
        for (const auto& c : platform.injectLibCandidates(exe)) {
            std::fprintf(stderr, "         %s\n", c.string().c_str());
        }
        return 3;
    }

    const std::vector<std::string> plan = resolvePassPlan(args);
    const bool multipass = plan.size() > 1;

    const std::string analysis_id = multipass ? makeAnalysisId() : std::string();
    const std::string dir_tag = multipass ? analysis_id : makeSessionId();

    const std::string app_name = args.name.empty()
                               ? platform.defaultAppName(args.command.front())
                               : args.name;

    // Where this run's session folder(s) land. The default output dir is already
    // per-run (defaultOutputDir(dir_tag)), so sessions sit flat inside it. An
    // explicit --output is a dir the user reuses across runs:
    //   - multi-pass nests its passes under a readable "run-<name>-<analysis_id>"
    //     folder so the passes of one analysis stay grouped (a flat dir would lose
    //     which sessions belong to the same run);
    //   - single-pass is one session, so it lands flat as <output>/<session_id>/ -
    //     matching embedded gpufl and the upload agent's <parent>/<session_id>
    //     discovery, with no redundant per-session wrapper.
    fs::path output_dir;
    if (args.output_dir.empty()) {
        output_dir = platform.defaultOutputDir(dir_tag);
    } else if (multipass) {
        const std::string run_folder =
            "run-" + slugForPath(app_name) + "-" + dir_tag.substr(0, 8);
        output_dir = fs::path(args.output_dir) / run_folder;
    } else {
        output_dir = fs::path(args.output_dir);
    }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::fprintf(stderr, "gpufl: cannot create %s: %s\n",
                     output_dir.string().c_str(), ec.message().c_str());
        return 2;
    }
    const fs::path agent_output_dir = fs::weakly_canonical(output_dir, ec);
    const fs::path upload_dir = ec ? output_dir : agent_output_dir;

    DebugLogger::setEnabled(args.verbose);

    std::string error;
    if (!platform.prepareInjectionEnv(inject_lib, error)) {
        std::fprintf(stderr, "gpufl: %s\n", error.c_str());
        return 2;
    }

    if (!setEnvOrPrint(platform, env::kCudaInjection64Path, inject_lib.string()) ||
        !setEnvOrPrint(platform, env::kNvtxInjection64Path, inject_lib.string()) ||
        !setEnvOrPrint(platform, env::kInject, "1") ||
        !setEnvOrPrint(platform, env::kAppName, app_name) ||
        !setEnvOrPrint(platform, env::kLogDir, output_dir.string()) ||
        !setEnvOrPrint(platform, env::kInjectProfile, inject::kProfileComprehensive) ||
        !setEnvOrPrint(platform, env::kInjectUpload, "0")) {
        return 2;
    }

    // Warmup defers the in-child capture start so the window skips cold-start
    // (autotune / JIT / allocator growth). Reuses the inject init-delay knob.
    if (args.warmup_ms > 0 &&
        !setEnvOrPrint(platform, env::kInjectInitDelayMs,
                       std::to_string(args.warmup_ms))) {
        return 2;
    }

    // --pc-sample-period: sample more frequently than the default so short
    // kernels that yield no samples produce data. 0 = leave the engine default;
    // the injected target reads GPUFL_PC_SAMPLING_PERIOD in gpufl::init().
    if (args.pc_sample_period != 0 &&
        !setEnvOrPrint(platform, env::kPcSamplingPeriod,
                       std::to_string(args.pc_sample_period))) {
        return 2;
    }

    // A bounded window stops the target after warmup+window wall-clock;
    // run_ms == 0 keeps the historical "run until the target exits" behavior.
    RunOptions run_opts;
    if (args.window_ms > 0) {
        run_opts.run_ms = args.warmup_ms + args.window_ms;
    }
    if (args.window_timeout_ms > 0) {
        run_opts.run_ms = run_opts.run_ms > 0
                        ? std::min(run_opts.run_ms, args.window_timeout_ms)
                        : args.window_timeout_ms;
    }

    AgentProcess agent;
    if (args.upload) {
        const fs::path cursor = args.agent_cursor.empty()
                              ? upload_dir / "cursor.json"
                              : fs::path(args.agent_cursor);

        AgentOptions agent_opts;
        agent_opts.source_folders = upload_dir.string();
        agent_opts.log_types = args.log_types;
        agent_opts.cursor_file = cursor.string();
        agent_opts.backend_url = resolveOption(args.backend_url, env::kBackendUrl);
        agent_opts.api_key = resolveOption(args.api_key, env::kApiKey);
        agent_opts.api_version = args.api_version;
        agent_opts.agent_jar = args.agent_jar;
        agent_opts.scope_to_new_sessions = true;  // upload only this run's session(s)

        if (!configureAgentEnvironment(agent_opts, error)) {
            std::fprintf(stderr, "gpufl trace --upload: %s\n", error.c_str());
            return 2;
        }

        AgentLaunchPlan agent_plan;
        if (!buildAgentLaunchPlan(agent_opts, agent_plan, error)) {
            std::fprintf(stderr, "gpufl trace --upload: %s\n", error.c_str());
            return 2;
        }
        if (!args.quiet) {
            std::fprintf(stderr, "[gpufl] starting agent: %s\n",
                         agent_plan.description.c_str());
        }
        if (!agent.start(agent_plan.command, error)) {
            std::fprintf(stderr, "gpufl trace --upload: %s\n", error.c_str());
            return 3;
        }
    }

    if (multipass) {
        if (!setEnvOrPrint(platform, env::kAnalysisId, analysis_id) ||
            !setEnvOrPrint(platform, env::kPassCount, std::to_string(plan.size()))) {
            return 2;
        }
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] capturing -> %s\n", output_dir.string().c_str());
        if (multipass) {
            std::fprintf(stderr, "[gpufl] multi-pass analysis %s - %zu passes:",
                         analysis_id.c_str(), plan.size());
            for (const auto& e : plan) std::fprintf(stderr, " %s", e.c_str());
            std::fputc('\n', stderr);
        }
        if (args.verbose) {
            std::fprintf(stderr, "[gpufl] inject lib: %s\n",
                         inject_lib.string().c_str());
            std::fprintf(stderr, "[gpufl] app_name: %s\n", app_name.c_str());
        }
    }

    int overall_rc = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const std::string& engine = plan[i];

        // A '+'-joined token runs its engines together in one process (combo);
        // a plain token is a single engine. Set exactly one path and clear the
        // other so a combo never leaks from a previous pass.
        const std::string combo = comboForToken(engine);
        if (combo.empty()) {
            if (!setEnvOrPrint(platform, env::kProfilingEngine, engine) ||
                !setEnvOrPrint(platform, env::kEngineCombo, std::string())) {
                return 2;
            }
        } else {
            if (!setEnvOrPrint(platform, env::kEngineCombo, combo) ||
                !setEnvOrPrint(platform, env::kProfilingEngine,
                               firstEngineOfToken(engine))) {
                return 2;
            }
        }
        if (multipass &&
            !setEnvOrPrint(platform, env::kPassIndex, std::to_string(i))) {
            return 2;
        }

        const std::string what =
            multipass ? "pass " + std::to_string(i + 1) + "/" +
                            std::to_string(plan.size()) + " (" + engine + ")"
                      : std::string("target");

        if (!args.quiet && multipass) {
            std::fprintf(stderr, "\n[gpufl] -- %s --\n", what.c_str());
            if (engine.find("PcSampling") != std::string::npos) {
                std::fprintf(stderr,
                    "[gpufl] note: PcSampling needs admin / NVIDIA CP \"allow GPU "
                    "performance counters to all users\"; this pass reports \"needs "
                    "admin\" and yields no PC data if unprivileged.\n");
            }
        }

        const int64_t pass_start_ns = nowNs();
        const auto t_start = std::chrono::steady_clock::now();
        const TraceProcessResult run = platform.runProcess(args.command, run_opts);
        const auto t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();

        if (run.launcher_error) {
            std::fprintf(stderr, "gpufl: %s\n", run.error.c_str());
            return run.rc;
        }

        if (!args.quiet) {
            if (run.window_stopped) {
                std::fprintf(stderr,
                             "[gpufl] %s window reached; stopped target in %.2fs\n",
                             what.c_str(), t_elapsed / 1000.0);
            } else if (run.signaled) {
                std::fprintf(stderr, "[gpufl] %s killed by signal %d in %.2fs\n",
                             what.c_str(), run.signal, t_elapsed / 1000.0);
            } else {
                std::fprintf(stderr, "[gpufl] %s exited (rc=%d) in %.2fs\n",
                             what.c_str(), run.rc, t_elapsed / 1000.0);
            }
        }

        // An intentional window stop is a successful capture, not a failed pass.
        if (!run.window_stopped && run.rc != 0 && overall_rc == 0) {
            overall_rc = run.rc;
        }

        // A child stopped ungracefully (window stop -> SIGKILL, or a crash)
        // never ran shutdown, so its logs are still the active files in
        // <session>/.tmp/ and there is no shutdown record. Publish .tmp into the
        // session root FIRST so the marker check can see job_start and synthesize
        // the missing shutdown; otherwise it scans an empty session dir and skips
        // it (the logs only surface in the post-loop repair, too late). Clean
        // exits already published + wrote a real shutdown, so this is a no-op
        // for them. The post-loop repair re-runs salvage harmlessly.
        (void)salvageSessionTempDirs(output_dir);

        SyntheticShutdownContext shutdown_context;
        shutdown_context.exit_code = run.rc;
        shutdown_context.signaled = run.signaled;
        shutdown_context.signal = run.signal;
        shutdown_context.window_stopped = run.window_stopped;
        const int synthesized = ensureTraceCompletionMarkers(
                output_dir, pass_start_ns, shutdown_context);
        if (synthesized > 0) {
            GFL_LOG_DEBUG("wrote ", synthesized, " synthetic shutdown marker(s)");
        }
    }

    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] inspect: %s\n", output_dir.string().c_str());
    }
    if (args.upload) {
        // The agent self-exits once it has uploaded every window of this session
        // (GPUFL_AGENT_EXIT_WHEN_DRAINED). Wait for that clean finish - up to
        // --agent-drain-ms as a safety cap - instead of a fixed sleep that used to
        // hard-kill the agent mid-upload and drop its late windows.
        const int cap = args.agent_drain_ms;
        GFL_LOG_DEBUG("waiting up to ", cap / 1000.0, "s for agent to finish uploading");
        if (agent.waitForExit(cap)) {
            GFL_LOG_DEBUG("agent finished uploading");
            // Clean drain: every window was 202-accepted, so no more chunks are
            // coming. Signal upload-complete from here (outside the agent's racing
            // shutdown) so the backend finalizes on its fast path.
            gpufl::BackendConfig config;
            config.backend_url = resolveOption(args.backend_url, env::kBackendUrl);
            config.api_key = resolveOption(args.api_key, env::kApiKey);
            config.api_path = args.api_version.empty() ? "" : "/api/" + args.api_version;
            signalSessionsComplete(output_dir, config, args.quiet);
        } else {
            if (!args.quiet) {
                GFL_LOG_ERROR("agent still running after ", cap / 1000.0,
                              "s cap - stopping (late windows may need a post-hoc `gpufl upload`) ");
            }
            agent.stop();
        }
    }

    const int repaired_logs = repairUncompressedLogs(output_dir);
    if (repaired_logs > 0) {
        GFL_LOG_DEBUG("compressed ", repaired_logs, " log file(s)");
    }

    return overall_rc;
}

}  // namespace gpufl::launcher
