// Implements `gpufl upload <log_path> [opts]`.
//
// Uploads a captured session directory by spawning the gpufl-agent - the single
// upload path. It speaks the same POST /api/v1/events/stream protocol the live
// `gpufl trace --upload` agent uses, with its cursor dedup, salvage, and
// session-complete handling, so there is no second uploader to keep in sync.
//
// (The in-process C++ gpufl::uploadLogs() is retained only for the embedded /
// no-Java path and is deprecated - see upload_logs.hpp.)

#include "upload_command.hpp"

#include "agent_launcher.hpp"
#include "gpufl/core/env_vars.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace gpufl::launcher {

namespace fs = std::filesystem;

namespace {

// Flag value if set, else the env var, else empty.
std::string resolve(const std::string& flag, const char* env_name) {
    if (!flag.empty()) return flag;
    if (const char* v = std::getenv(env_name); v && *v) return v;
    return {};
}

// Agent api-version token ("v1") from a --api-path mount ("/api/v1"); the agent
// rebuilds "/api/<version>/events/stream" itself. Empty path -> "v1".
std::string apiVersionFrom(const std::string& api_path) {
    std::string s = api_path;
    while (!s.empty() && s.back() == '/') s.pop_back();
    if (s.empty()) return "v1";
    const auto pos = s.find_last_of('/');
    std::string seg = (pos == std::string::npos) ? s : s.substr(pos + 1);
    return seg.empty() ? "v1" : seg;
}

}  // namespace

int runUpload(const UploadArgs& args) {
    const std::string backend_url = resolve(args.backend_url, gpufl::env::kBackendUrl);
    const std::string api_key     = resolve(args.api_key,     gpufl::env::kApiKey);

    if (backend_url.empty()) {
        std::fprintf(stderr,
            "gpufl upload: --backend-url required (or set GPUFL_BACKEND_URL)\n");
        return 2;
    }
    if (api_key.empty()) {
        std::fprintf(stderr,
            "gpufl upload: --api-key required (or set GPUFL_API_KEY)\n");
        return 2;
    }
    std::error_code ec;
    if (!fs::is_directory(args.log_path, ec)) {
        std::fprintf(stderr, "gpufl upload: not a directory: %s\n", args.log_path.c_str());
        return 2;
    }

    // --force re-uploads every session. Two things gate a re-upload: the agent's
    // per-session settle marker (discovery skips a marked session) and the cursor
    // (skips shipped windows). So clear the markers AND use a throwaway cursor; the
    // backend still de-duplicates on ingest. Normal runs keep a persistent cursor in
    // the log dir and leave markers, so re-runs are cheap no-ops.
    std::string cursor;
    if (args.force) {
        // Marker names mirror the agent's SessionWatcher.{UPLOADED,FAILED}_MARKER.
        std::error_code walk_ec;
        for (fs::recursive_directory_iterator it(args.log_path, walk_ec), end;
             it != end; it.increment(walk_ec)) {
            if (walk_ec) break;
            const std::string name = it->path().filename().string();
            if (name == ".uploaded" || name == ".failed") {
                std::error_code rm_ec;
                fs::remove(it->path(), rm_ec);
            }
        }
        fs::path tmp = fs::temp_directory_path() / "gpufl-upload-force-cursor.json";
        fs::remove(tmp, ec);
        cursor = tmp.string();
    } else {
        cursor = (fs::path(args.log_path) / "cursor.json").string();
    }

    AgentOptions agent_opts;
    agent_opts.source_folders = args.log_path;
    agent_opts.log_types      = "device,scope,system,sass";
    agent_opts.backend_url    = backend_url;
    agent_opts.api_key        = api_key;
    agent_opts.api_version    = apiVersionFrom(resolve(args.api_path, gpufl::env::kApiPath));
    agent_opts.agent_jar      = args.agent_jar;
    agent_opts.cursor_file    = cursor;
    agent_opts.exit_if_empty  = true;  // one-shot: nothing to upload -> exit, don't wait
    // scope_to_new_sessions stays false: `gpufl upload` ships every session in the dir.

    std::string error;
    if (!configureAgentEnvironment(agent_opts, error)) {
        std::fprintf(stderr, "gpufl upload: %s\n", error.c_str());
        return 2;
    }
    AgentLaunchPlan plan;
    if (!buildAgentLaunchPlan(agent_opts, plan, error)) {
        std::fprintf(stderr, "gpufl upload: %s\n", error.c_str());
        return 2;
    }
    if (!args.quiet) {
        std::fprintf(stderr, "[gpufl] uploading %s via %s\n",
                     args.log_path.c_str(), plan.description.c_str());
    }

    AgentProcess agent;
    if (!agent.start(plan.command, error)) {
        std::fprintf(stderr, "gpufl upload: %s\n", error.c_str());
        return 3;
    }
    const int cap_ms = args.timeout_s > 0 ? args.timeout_s * 1000 : 300000;
    if (agent.waitForExit(cap_ms)) {
        if (!args.quiet) std::printf("Upload complete.\n");
        return 0;
    }
    agent.stop();
    std::fprintf(stderr,
        "gpufl upload: agent still running after %ds - stopped; re-run to resume\n",
        args.timeout_s);
    return 1;
}

}  // namespace gpufl::launcher
