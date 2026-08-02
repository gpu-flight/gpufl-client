#include "trace_run_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

namespace gpufl::launcher {
namespace {

std::string makeSessionId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count())};
    const uint64_t value = rng();
    std::ostringstream out;
    out << std::hex << std::setw(8) << std::setfill('0')
        << (value & 0xffffffffu);
    return out.str();
}

std::string makeAnalysisId() {
    static std::mt19937_64 rng{
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        0x9e3779b97f4a7c15ULL};
    const uint64_t a = rng();
    const uint64_t b = rng();
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%08x-%04x-%04x-%04x-%012llx",
                  static_cast<unsigned>(a >> 32),
                  static_cast<unsigned>((a >> 16) & 0xffff),
                  static_cast<unsigned>(a & 0xffff),
                  static_cast<unsigned>((b >> 48) & 0xffff),
                  static_cast<unsigned long long>(b & 0xffffffffffffULL));
    return buffer;
}

// Filesystem-safe slug for a multi-pass run-folder name. Empty input falls
// back to "session" so planning never creates an invalid or invisible folder.
std::string slugForPath(const std::string& input) {
    const auto safe = [](const char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
    };
    std::string output;
    for (const char c : input) output.push_back(safe(c) ? c : '-');
    while (!output.empty() && output.front() == '-') output.erase(output.begin());
    while (!output.empty() && output.back() == '-') output.pop_back();
    if (output.size() > 48) output.resize(48);
    return output.empty() ? std::string("session") : output;
}

}  // namespace

TraceRunPlan createTraceRunPlan(const TraceArgs& args,
                                const TracePlatform& platform) {
    TraceRunPlan plan;
    plan.passes = resolvePassPlan(args);
    plan.multipass = plan.passes.size() > 1;
    plan.segmented = segmentationRequested(args);
    plan.analysis_id = plan.multipass ? makeAnalysisId() : std::string();
    plan.run_id = plan.segmented ? generateRunId() : std::string();
    plan.directory_tag = plan.multipass
        ? plan.analysis_id
        : (plan.segmented ? plan.run_id : makeSessionId());
    plan.app_name = args.name.empty()
        ? platform.defaultAppName(args.command.front())
        : args.name;

    if (args.output_dir.empty()) {
        plan.output_dir = platform.defaultOutputDir(plan.directory_tag);
    } else if (plan.multipass) {
        const std::string run_folder =
            "run-" + slugForPath(plan.app_name) + "-" +
            plan.directory_tag.substr(0, 8);
        plan.output_dir = fs::path(args.output_dir) / run_folder;
    } else {
        plan.output_dir = fs::path(args.output_dir);
    }

    if (args.window_ms > 0) {
        plan.run_options.run_ms = args.warmup_ms + args.window_ms;
    }
    if (args.window_timeout_ms > 0) {
        plan.run_options.run_ms = plan.run_options.run_ms > 0
            ? std::min(plan.run_options.run_ms, args.window_timeout_ms)
            : args.window_timeout_ms;
    }
    return plan;
}

}  // namespace gpufl::launcher
