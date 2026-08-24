#include "cli_trace_options.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cli_option_manager.hpp"
#include "gpufl/core/segmentation_config.hpp"

namespace gpufl::launcher {

namespace {

using detail::CliOptionManager;
using detail::FlagBreak;

constexpr int kSection(const TraceHelpSection section) {
    return static_cast<int>(section);
}

// ── Generic handlers ──────────────────────────────────────────────────────

template <std::string TraceArgs::*Slot>
std::string parseStringOption(const FlagBreak& flag,
                              const std::vector<std::string>& argv,
                              std::size_t& index,
                              TraceArgs& args) {
    return detail::takeFlagValue(flag, argv, index, args.*Slot);
}

template <std::string TraceArgs::*Slot>
std::string parseNonEmptyStringOption(const FlagBreak& flag,
                                      const std::vector<std::string>& argv,
                                      std::size_t& index,
                                      TraceArgs& args) {
    std::string& value = args.*Slot;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    return value.empty() ? flag.key + " cannot be empty" : std::string{};
}

template <bool TraceArgs::*Slot>
std::string setFlag(const FlagBreak& flag,
                    const std::vector<std::string>&,
                    std::size_t&,
                    TraceArgs& args) {
    // A boolean flag with `=value` is a misunderstanding, not a value: report it
    // the same way an unknown flag is reported rather than silently ignoring it.
    if (flag.inline_value) return "unknown flag: " + flag.key;
    args.*Slot = true;
    return {};
}

/** Duration into a slot, with the shared "expected a duration" message. */
template <std::int64_t TraceArgs::*Slot>
std::string parseDurationOption(const FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index,
                                TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    if (!detail::parseDurationMs(value, args.*Slot)) {
        return "invalid " + flag.key + " value: " + value +
               " (expected a duration like 30s, 500ms, 5m, 1h, or a bare "
               "number of seconds)";
    }
    return {};
}

/** Non-negative int into a slot (milliseconds today). */
template <int TraceArgs::*Slot>
std::string parseNonNegativeIntOption(const FlagBreak& flag,
                                      const std::vector<std::string>& argv,
                                      std::size_t& index,
                                      TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    if (!detail::parseNonNegativeInt(value, args.*Slot)) {
        return "invalid " + flag.key + " value: " + value +
               " (expected a non-negative integer, milliseconds)";
    }
    return {};
}

/** uint64 into a slot; MinValue > 0 rejects zero as well as garbage. */
template <std::uint64_t TraceArgs::*Slot, std::uint64_t MinValue>
std::string parseUint64Option(const FlagBreak& flag,
                              const std::vector<std::string>& argv,
                              std::size_t& index,
                              TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    bool ok = detail::parseUint64(value, args.*Slot);
    // if constexpr: `unsigned < 0` is a tautology the compiler warns about.
    if constexpr (MinValue > 0) {
        ok = ok && args.*Slot >= MinValue;
    }
    if (!ok) {
        if constexpr (MinValue > 0) {
            return "invalid " + flag.key + " value: " + value +
                   " (expected a positive integer)";
        } else {
            return "invalid " + flag.key + " value: " + value +
                   " (expected a non-negative integer; 0 disables it)";
        }
    }
    return {};
}

/** Byte budget into a slot; 0 disables the trigger. */
template <std::uint64_t TraceArgs::*Slot>
std::string parseByteSizeOption(const FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index,
                                TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
        }
    if (!detail::parseByteSize(value, args.*Slot)) {
        return "invalid " + flag.key + " value: " + value +
               " (expected a byte count with an optional k/m/g suffix, "
               "e.g. 50g; 0 disables it)";
    }
    return {};
}

// A flag that no longer exists: consume its value so the next token is not
// mistaken for a command, then explain where it went. Two named handlers rather
// than a template over the message: taking the address of an internal-linkage
// constant as a template argument is portable in theory and fussy in practice.
std::string consumeRemovedValue(const FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index) {
    std::string ignored;
    return detail::takeFlagValue(flag, argv, index, ignored);
}

std::string rejectProfileOption(const FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index,
                                TraceArgs&) {
    if (const std::string error = consumeRemovedValue(flag, argv, index);
        !error.empty()) {
        return error;
    }
    return "`gpufl trace --profile` has been removed; use --passes=Trace, "
           "--passes=Deep, or `gpufl monitor` for monitoring-only telemetry";
}

std::string rejectEngineOption(const FlagBreak& flag,
                               const std::vector<std::string>& argv,
                               std::size_t& index,
                               TraceArgs&) {
    if (const std::string error = consumeRemovedValue(flag, argv, index);
        !error.empty()) {
        return error;
    }
    return "`gpufl trace --engine` has been removed; use --passes=Trace, "
           "--passes=Deep, or an explicit list like --passes=Trace,PmSampling";
}

// ── --passes ──────────────────────────────────────────────────────────────

// Canonical engine names accepted by each --passes token. Must match the set
// gpufl::init() parses for GPUFL_PROFILING_ENGINE (see gpufl.cpp). The launcher
// only validates + forwards verbatim; init() is the single string->enum parser,
// so this is the one launcher-side copy to keep in sync with the ladder.
constexpr const char* kEngines[] = {
    "Trace", "PcSampling", "SassMetrics",
    "PmSampling", "RangeProfiler", "RangeProfilerKernelReplay", "Deep"};

bool isValidEngine(const std::string& engine) {
    for (const char* known : kEngines) {
        if (engine == known) return true;
    }
    return false;
}

// A token may be a single engine ("Trace") or a '+'-joined composite group
// ("Trace+PcSampling") that runs those engines together in ONE process.
std::string validatePassToken(const std::string& token) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t plus = token.find('+', start);
        parts.push_back(detail::trim(token.substr(
            start, plus == std::string::npos ? std::string::npos : plus - start)));
        if (plus == std::string::npos) break;
        start = plus + 1;
    }
    const bool composite = parts.size() > 1;
    for (const std::string& part : parts) {
        if (part.empty()) {
            return "empty engine in --passes group '" + token +
                   "' (expected e.g. Trace+PcSampling)";
        }
        if (!isValidEngine(part)) {
            return "invalid --passes engine: " + part +
                   " (expected a comma-separated list of: Trace | PcSampling | "
                   "SassMetrics | PmSampling | RangeProfiler | "
                   "RangeProfilerKernelReplay | Deep; join engines with + to run "
                   "them in one process, e.g. Trace+PcSampling)";
        }
        if (composite && part == "Deep") {
            return "Deep cannot be combined in a '+' group (it already runs "
                   "PcSampling + SassMetrics together); give it its own pass";
        }
        if (composite && part == "SassMetrics") {
            return "SassMetrics cannot share a process (it deadlocks with kernel "
                   "tracing); give it its own pass with a comma, e.g. "
                   "Trace+PcSampling,SassMetrics";
        }
    }
    return {};
}

std::string parsePasses(const FlagBreak& flag,
                        const std::vector<std::string>& argv,
                        std::size_t& index,
                        TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    args.passes.clear();
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = value.find(',', start);
        const std::string item = detail::trim(value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!item.empty()) {
            if (const std::string error = validatePassToken(item);
                !error.empty()) {
                return error;
            }
            args.passes.push_back(item);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (args.passes.empty()) {
        return "--passes requires at least one engine";
    }
    return {};
}

// ── Segmentation, window, deep, sampling ──────────────────────────────────

std::string parseSegmentEvery(const FlagBreak& flag,
                              const std::vector<std::string>& argv,
                              std::size_t& index,
                              TraceArgs& args) {
    if (const std::string error =
            parseDurationOption<&TraceArgs::segment_every_ms>(
                flag, argv, index, args);
        !error.empty()) {
        return error;
    }
    if (args.segment_every_ms > 0 &&
        args.segment_every_ms < kMinSegmentEveryMs) {
        return "--segment-every must be at least 60s; shorter cadences can "
               "create a session storm";
    }
    return {};
}

std::string parseAfterWindow(const FlagBreak& flag,
                             const std::vector<std::string>& argv,
                             std::size_t& index,
                             TraceArgs& args) {
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, args.after_window);
        !error.empty()) {
        return error;
    }
    if (args.after_window == "keep") {
        return "--after-window=keep is not yet implemented; the launcher stops "
               "the target at window end (restart it with a script)";
    }
    if (args.after_window != "stop") {
        return "invalid --after-window value: " + args.after_window +
               " (expected: stop)";
    }
    return {};
}

// One handler for the three deep durations: they share the unit-less-seconds
// trap and the deep_requested side effect, and flag.key says which one fired.
std::string parseDeepDuration(const FlagBreak& flag,
                              const std::vector<std::string>& argv,
                              std::size_t& index,
                              TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    std::int64_t ms = 0;
    if (!detail::parseDurationMs(value, ms)) {
        return "invalid " + flag.key + " value: " + value +
               " (expected a duration like 30s, 500ms, 5m, 1h, or a bare "
               "number of seconds)";
    }
    // A bare number is seconds, which collides with --deep-launches:
    // `--deep-for=2000` meaning "2000 launches" silently becomes a 33-minute
    // window, and the no-bound check cannot catch it because a bound *was*
    // given. Reject the unit-less form once it is too large to plausibly be a
    // duration someone typed on purpose - naming the alternative, since that is
    // the mistake.
    const bool unitless =
        !value.empty() &&
        value.find_first_not_of("0123456789") == std::string::npos;
    if (flag.key == "--deep-for" && unitless && ms >= 600'000) {
        return "--deep-for=" + value + " means " + std::to_string(ms / 1000) +
               " SECONDS (a bare number is seconds), which is almost certainly "
               "not what you meant. Add a unit (e.g. " + value +
               "s, 5m) if you really want that long a window, or use "
               "--deep-launches " + value + " to bound it by kernel launches "
               "instead";
    }
    if (flag.key == "--deep-after") {
        args.deep_after_ms = ms;
        args.deep_after_set = true;
    } else if (flag.key == "--deep-for") {
        args.deep_for_ms = ms;
    } else {
        args.deep_cooldown_ms = ms;
    }
    args.deep_requested = true;
    return {};
}

std::string parseDeepWhen(const FlagBreak& flag,
                          const std::vector<std::string>& argv,
                          std::size_t& index,
                          TraceArgs& args) {
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, args.deep_when);
        !error.empty()) {
        return error;
    }
    if (args.deep_when.empty()) {
        return "--deep-when needs an expression, e.g. "
               "--deep-when=\"custom.token_rate<1000 for 2s\"";
    }
    // Parsed by the client, not here: telling a misspelled built-in from a
    // custom counter that has simply not registered yet needs the metric
    // registry, and duplicating that check would give two places to disagree
    // about what a metric name means.
    args.deep_requested = true;
    return {};
}

std::string parseDeepLaunches(const FlagBreak& flag,
                              const std::vector<std::string>& argv,
                              std::size_t& index,
                              TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    if (!detail::parseUint64(value, args.deep_launches) ||
        args.deep_launches == 0) {
        return "invalid --deep-launches value: " + value +
               " (expected a positive number of kernel launches)";
    }
    args.deep_requested = true;
    return {};
}

std::string parsePcSamplePeriod(const FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index,
                                TraceArgs& args) {
    std::string value;
    if (const std::string error =
            detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    int period = 0;
    if (!detail::parseNonNegativeInt(value, period) || period < 5 ||
        period > 31) {
        return "invalid --pc-sample-period value: " + value +
               " (expected an integer 5..31: log2 of GPU cycles per sample)";
    }
    args.pc_sample_period = static_cast<std::uint32_t>(period);
    return {};
}

// ── The table ─────────────────────────────────────────────────────────────

const CliOptionManager<TraceArgs>& traceOptions() {
    static const CliOptionManager<TraceArgs> manager = [] {
        CliOptionManager<TraceArgs> options;
        options
            // Capture
            .add({"-n", "--name"}, "<NAME>",
                 "Session name (default: basename of <COMMAND>)",
                 kSection(TraceHelpSection::Capture),
                 &parseStringOption<&TraceArgs::name>)
            .add({"-o", "--output"}, "<DIR>",
                 "Local NDJSON output dir (default: "
                 "~/.gpufl/traces/{ts}_{session_id}/)",
                 kSection(TraceHelpSection::Capture),
                 &parseStringOption<&TraceArgs::output_dir>)
            .add({"--passes"}, "<LIST>",
                 "Capture pass list: comma-separated values from Trace | "
                 "PcSampling | SassMetrics | PmSampling | RangeProfiler | "
                 "RangeProfilerKernelReplay | Deep. Each comma is a separate "
                 "pass (relaunch). Join engines with + to run them in ONE "
                 "process, e.g. Trace+PcSampling (timeline + PC stalls, one "
                 "run). Default: Trace. Deep runs PcSampling+SassMetrics in one "
                 "pass (same as the embedded Deep engine); for "
                 "timeline+stalls+SASS list passes explicitly, e.g. "
                 "Trace,PcSampling,SassMetrics. SassMetrics must be its own "
                 "pass (deadlocks if shared). Use gpufl monitor for "
                 "monitoring-only telemetry. PcSampling / PM / Range passes may "
                 "need NVIDIA performance-counter access.",
                 kSection(TraceHelpSection::Capture), &parsePasses)
            .add({"--no-source"}, "",
                 "Do not capture discovered CUDA source file content. Source "
                 "capture is enabled by default for analysis and source-line "
                 "correlation",
                 kSection(TraceHelpSection::Capture),
                 &setFlag<&TraceArgs::no_source>)
            .add({"--source-root"}, "<DIR>",
                 "Allow correlated CUDA/C++ source only within this project "
                 "directory (default: current working directory)",
                 kSection(TraceHelpSection::Capture),
                 &parseNonEmptyStringOption<&TraceArgs::source_root>)

            // Runtime
            .add({"-q", "--quiet"}, "",
                 "Suppress launcher chatter (errors still print)",
                 kSection(TraceHelpSection::Runtime),
                 &setFlag<&TraceArgs::quiet>)
            .add({"-v", "--verbose"}, "", "Verbose launcher logging",
                 kSection(TraceHelpSection::Runtime),
                 &setFlag<&TraceArgs::verbose>)
            .add({"--upload"}, "", "Start gpufl-agent as the live uploader",
                 kSection(TraceHelpSection::Runtime),
                 &setFlag<&TraceArgs::upload>)
            .add({"--backend-url"}, "<URL>",
                 "Backend base URL for --upload (env: GPUFL_BACKEND_URL)",
                 kSection(TraceHelpSection::Runtime),
                 &parseStringOption<&TraceArgs::backend_url>)
            .add({"--api-key"}, "<KEY>",
                 "Bearer token for --upload (env: GPUFL_API_KEY)",
                 kSection(TraceHelpSection::Runtime),
                 &parseStringOption<&TraceArgs::api_key>)
            .add({"--api-version"}, "<VER>",
                 "Agent HTTP API version (default: v1)",
                 kSection(TraceHelpSection::Runtime),
                 &parseNonEmptyStringOption<&TraceArgs::api_version>)
            .add({"--agent-jar"}, "<PATH>",
                 "Run agent as java -jar <PATH> (env: GPUFL_AGENT_JAR)",
                 kSection(TraceHelpSection::Runtime),
                 &parseNonEmptyStringOption<&TraceArgs::agent_jar>)
            .add({"--agent-cursor"}, "<PATH>",
                 "Agent cursor file (default: <output>/cursor.json)",
                 kSection(TraceHelpSection::Runtime),
                 &parseNonEmptyStringOption<&TraceArgs::agent_cursor>)
            .add({"--log-types"}, "<LIST>",
                 "Agent channels to upload (default: device,scope,system,sass)",
                 kSection(TraceHelpSection::Runtime),
                 &parseNonEmptyStringOption<&TraceArgs::log_types>)
            .add({"--agent-drain-ms"}, "<MS>",
                 "Maximum wait for the agent to finish uploading "
                 "(default: 60000)",
                 kSection(TraceHelpSection::Runtime),
                 &parseNonNegativeIntOption<&TraceArgs::agent_drain_ms>)

            // Segmentation
            .add({"--segment-every"}, "<DUR>",
                 "Split a long run on this cadence (minimum: 60s). Example: "
                 "--segment-every=5m. Default: off",
                 kSection(TraceHelpSection::Segmentation), &parseSegmentEvery)
            .add({"--segment-max-rows"}, "<N>",
                 "Also split after this many logical telemetry rows. The batch "
                 "crossing N stays in the old segment. Default: off. V1 "
                 "supports one Trace or PM pass; multi-pass analyses cannot be "
                 "segmented.",
                 kSection(TraceHelpSection::Segmentation),
                 &parseUint64Option<&TraceArgs::segment_max_rows, 0>)
            .add({"--roll-every"}, "<DUR>",
                 "End the run and start a new part after this long, restarting "
                 "at segment 0. Requires --segment-every: the part ends at the "
                 "next segment boundary. Default: off",
                 kSection(TraceHelpSection::Segmentation),
                 &parseDurationOption<&TraceArgs::run_roll_every_ms>)
            .add({"--roll-max-bytes"}, "<SIZE>",
                 "Also end the run part after this much serialized telemetry, "
                 "e.g. 50g. Counts every channel across the whole part, not "
                 "per segment. Default: off",
                 kSection(TraceHelpSection::Segmentation),
                 &parseByteSizeOption<&TraceArgs::run_roll_max_bytes>)
            // Window
            .add({"--warmup"}, "<DUR>",
                 "Skip cold start: defer capture by this long (e.g. 30s, "
                 "500ms, 5m; bare number = seconds)",
                 kSection(TraceHelpSection::Window),
                 &parseDurationOption<&TraceArgs::warmup_ms>)
            .add({"--window"}, "<DUR>",
                 "Bounded window: capture this long after warmup, then STOP "
                 "the target. For servers that never exit. Omit to run to the "
                 "target's own exit.",
                 kSection(TraceHelpSection::Window),
                 &parseDurationOption<&TraceArgs::window_ms>)
            .add({"--window-timeout"}, "<DUR>",
                 "Hard cap on total target runtime (safety).",
                 kSection(TraceHelpSection::Window),
                 &parseDurationOption<&TraceArgs::window_timeout_ms>)
            .add({"--after-window"}, "<WHAT>",
                 "What to do at window end. Only 'stop' today.",
                 kSection(TraceHelpSection::Window), &parseAfterWindow)

            // Deep window
            .add({"--deep-after"}, "<DUR>",
                 "Arm the deep engine this long into the run, then disarm. "
                 "Unlike --window the target keeps running. Needs a bound "
                 "below. Default: 0 (arm at the first kernel launch).",
                 kSection(TraceHelpSection::Deep), &parseDeepDuration)
            .add({"--deep-for"}, "<DUR>",
                 "How long the deep window stays armed. Note this bounds TIME, "
                 "which does not bound how much the engines actually collect - "
                 "see --deep-launches.",
                 kSection(TraceHelpSection::Deep), &parseDeepDuration)
            .add({"--deep-when"}, "<EXPR>",
                 "Open the window when a metric crosses a threshold, e.g. "
                 "\"custom.token_rate<1000 for 2s\". This and --deep-after are "
                 "two answers to the same question, so pass one or the other.",
                 kSection(TraceHelpSection::Deep), &parseDeepWhen)
            .add({"--deep-launches"}, "<N>",
                 "Kernel-launch bound on the deep window; ends it at whichever "
                 "bound is hit first. PREFER THIS: wall time does not bound how "
                 "much an engine collects, and how far one second of it goes "
                 "differs by more than an order of magnitude between engines.",
                 kSection(TraceHelpSection::Deep), &parseDeepLaunches)
            .add({"--deep-cooldown"}, "<DUR>",
                 "Quiet time before another window may open.",
                 kSection(TraceHelpSection::Deep), &parseDeepDuration)

            // Sampling
            .add({"--pc-sample-period"}, "<N>",
                 "PC sampling period: log2 of GPU cycles per sample (5..31; "
                 "default 10). Lower = more frequent - for short kernels that "
                 "yield no PC samples by default.",
                 kSection(TraceHelpSection::Sampling), &parsePcSamplePeriod)

            // Removed flags: dispatched so the migration hint prints, but
            // deliberately undocumented (no description => absent from help).
            .add({"--profile"}, &rejectProfileOption)
            .add({"--engine"}, &rejectEngineOption);
        return options;
    }();
    return manager;
}

}  // namespace

TraceSimpleOptionResult parseTraceSimpleOption(
    const detail::FlagBreak& flag,
    const std::vector<std::string>& argv,
    std::size_t& index,
    TraceArgs& args) {
    std::string error;
    const bool found =
        traceOptions().parse(flag, argv, index, args, error);
    return {found, std::move(error)};
}

std::string formatTraceSimpleOptions(const TraceHelpSection section) {
    return traceOptions().formatHelp(kSection(section));
}

std::vector<std::string> traceOptionAliases() {
    std::vector<std::string> out;
    for (const std::string_view alias : traceOptions().aliases()) {
        out.emplace_back(alias);
    }
    return out;
}

}  // namespace gpufl::launcher
