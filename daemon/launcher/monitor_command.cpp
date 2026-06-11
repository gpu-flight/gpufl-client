#include "monitor_command.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

#include "agent_launcher.hpp"
#include "gpufl/core/env_vars.hpp"
#include "monitor_runner.hpp"

namespace fs = std::filesystem;

namespace gpufl::launcher {
namespace {

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        0x6eed0e9da4d94a4fULL};
    const uint64_t v = rng();
    std::ostringstream os;
    os << std::hex << std::setw(8) << std::setfill('0') << (v & 0xffffffffu);
    return os.str();
}

std::string makeTimestamp() {
    const auto t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

fs::path homeDir() {
#ifdef _WIN32
    if (const char* up = std::getenv("USERPROFILE"); up && *up) {
        return fs::path(up);
    }
    if (const char* hd = std::getenv("HOMEDRIVE")) {
        if (const char* hp = std::getenv("HOMEPATH")) {
            return fs::path(std::string(hd) + hp);
        }
    }
    return fs::temp_directory_path();
#else
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home);
    }
    return fs::path("/tmp");
#endif
}

fs::path defaultOutputDir() {
    return homeDir() / ".gpufl" / "monitor" /
           (makeTimestamp() + "_" + makeSessionId());
}

}  // namespace

int runMonitor(const MonitorArgs& args) {
    fs::path output_dir = args.output_dir.empty()
                        ? defaultOutputDir()
                        : fs::path(args.output_dir);
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        std::fprintf(stderr, "gpufl monitor: cannot create %s: %s\n",
                     output_dir.string().c_str(), ec.message().c_str());
        return 2;
    }
    output_dir = fs::weakly_canonical(output_dir, ec);

    AgentProcess agent;
    if (args.upload) {
        fs::path cursor = args.agent_cursor.empty()
                        ? output_dir / "cursor.json"
                        : fs::path(args.agent_cursor);

        AgentOptions agent_opts;
        agent_opts.source_folders = output_dir.string();
        agent_opts.log_types = args.log_types;
        agent_opts.cursor_file = cursor.string();
        agent_opts.backend_url = resolveOption(args.backend_url, gpufl::env::kBackendUrl);
        agent_opts.api_key = resolveOption(args.api_key, gpufl::env::kApiKey);
        agent_opts.api_version = args.api_version;
        agent_opts.agent_jar = args.agent_jar;

        std::string error;
        if (!configureAgentEnvironment(agent_opts, error)) {
            std::fprintf(stderr, "gpufl monitor --upload: %s\n", error.c_str());
            return 2;
        }

        AgentLaunchPlan plan;
        if (!buildAgentLaunchPlan(agent_opts, plan, error)) {
            std::fprintf(stderr, "gpufl monitor --upload: %s\n", error.c_str());
            return 2;
        }
        if (!args.quiet) {
            std::fprintf(stderr, "[gpufl] starting agent: %s\n",
                         plan.description.c_str());
        }
        if (!agent.start(plan.command, error)) {
            std::fprintf(stderr, "gpufl monitor --upload: %s\n", error.c_str());
            return 3;
        }
    }

    gpufl::daemon::MonitorRunOptions opts;
    opts.app_name = args.name;
    opts.log_path = output_dir.string();
    opts.interval_ms = args.interval_ms;
    opts.quiet = args.quiet;

    const int rc = gpufl::daemon::runMonitorForeground(opts);
    agent.stop();
    return rc;
}

}  // namespace gpufl::launcher
