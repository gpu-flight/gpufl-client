#include "cli_parse.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

#include "cli_parse_internal.hpp"
#include "cli_subcommand_options.hpp"
#include "cli_trace_options.hpp"
#include "gpufl/core/segmentation_config.hpp"

namespace gpufl::launcher {

namespace {

using detail::parseDurationMs;
using detail::parseNonNegativeInt;
using detail::parseUint64;
using detail::splitFlag;
using detail::takeFlagValue;
using detail::trim;

}  // namespace

ParsedTopLevel parseTopLevel(int argc, char** argv) {
    ParsedTopLevel out;
    if (argc < 2) {
        out.sub = Subcommand::Help;
        return out;
    }
    std::string first = argv[1];
    if (first == "-h" || first == "--help") {
        out.sub = Subcommand::Help;
    } else if (first == "-V" || first == "--version" || first == "version") {
        out.sub = Subcommand::Version;
    } else if (first == "trace") {
        out.sub = Subcommand::Trace;
    } else if (first == "upload") {
        out.sub = Subcommand::Upload;
    } else if (first == "monitor") {
        out.sub = Subcommand::Monitor;
    } else if (first == "info") {
        out.sub = Subcommand::Info;
    } else {
        out.sub = Subcommand::Unknown;
        out.remaining.push_back(first);
        return out;
    }
    for (int i = 2; i < argc; ++i) out.remaining.emplace_back(argv[i]);
    return out;
}

TraceParseResult parseTraceArgs(const std::vector<std::string>& argv) {
    TraceArgs out;
    bool seen_dash_dash = false;

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (!seen_dash_dash && tok == "--") {
            seen_dash_dash = true;
            continue;
        }
        if (seen_dash_dash) {
            out.command.push_back(tok);
            continue;
        }
        if (tok == "-h" || tok == "--help") {
            // Caller prints help; signal via empty error + no command.
            return {std::nullopt, "__help__"};
        }

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;

        const TraceSimpleOptionResult simple =
            parseTraceSimpleOption(fb, argv, i, out);
        if (simple.found) {
            if (!simple.error.empty()) return {std::nullopt, simple.error};
            continue;
        }

        // Not an option the registry knows. A non-flag token before `--` is
        // almost certainly the caller forgetting the splitter, e.g.
        // `gpufl trace python train.py`; distinguish that from a flag typo.
        if (!tok.empty() && tok[0] != '-') {
            return {std::nullopt, "missing `--` separator before command"};
        }
        return {std::nullopt, "unknown flag: " + key};
    }
    if (!seen_dash_dash) {
        return {std::nullopt, "missing `--` separator before command"};
    }
    if (out.command.empty()) {
        return {std::nullopt, "no command specified after `--`"};
    }
    // Explicit passes and deep windows are different execution models, and
    // mixing them cannot mean anything coherent: the deep engines have to be
    // fixed before the first CUDA call, so a --passes list either already
    // contains what the window would arm (making the flag redundant) or does
    // not (making the window arm nothing - which is what
    // `--passes=Trace --deep-after=30s` silently did). Rejected before the
    // target is launched, whichever order the flags came in.
    if (const std::string mode_error = validateTraceExecutionMode(out);
        !mode_error.empty()) {
        return {std::nullopt, mode_error};
    }

    // Two answers to "when does the window open" is one too many. Measured on
    // the 3090: with both set, the scheduled window opens at t=0, the rule's
    // request is refused as busy, and the rule then waits for a rearm that a
    // still-busy workload never gives - it reported `never_true` for a
    // condition that held the whole run.
    if (!out.deep_when.empty() && out.deep_after_set) {
        return {std::nullopt,
                "--deep-when and --deep-after are two different triggers for "
                "the same window. Pass --deep-when to open it on a metric, or "
                "--deep-after to open it at a fixed time"};
    }

    // A deep window with neither bound would arm and never disarm, which is
    // just "profile deeply for the whole run" with extra steps.
    if (out.deep_requested && out.deep_for_ms == 0 && out.deep_launches == 0) {
        return {std::nullopt,
                "a deep window needs a bound: pass --deep-launches <n> or "
                "--deep-for <duration> (prefer --deep-launches: it is what "
                "the engines actually scale with. The replay engines cover "
                "far less work per second of wall time, and PC sampling "
                "returns nothing at all below a few thousand launches)"};
    }
    return {out, ""};
}

UploadParseResult parseUploadArgs(const std::vector<std::string>& argv) {
    UploadArgs out;
    bool have_log_path = false;

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") return {std::nullopt, "__help__"};

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;
        const SubcommandOptionResult simple =
            parseUploadSimpleOption(fb, argv, i, out);
        if (simple.found) {
            if (!simple.error.empty()) return {std::nullopt, simple.error};
            continue;
        }

        if (key == "--session-id") {
            return {std::nullopt,
                    "--session-id is no longer supported; point <LOG_PATH> at a "
                    "directory containing only that session"};
        }
        if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + key};
        }
        // Bare token → the positional <LOG_PATH>. Only one allowed.
        if (have_log_path) {
            return {std::nullopt, "unexpected extra argument: " + tok +
                                      " (only one <LOG_PATH> is accepted)"};
        }
        out.log_path = tok;
        have_log_path = true;
    }

    if (!have_log_path) {
        return {std::nullopt, "missing <LOG_PATH> (the trace output directory)"};
    }
    return {out, ""};
}

MonitorParseResult parseMonitorArgs(const std::vector<std::string>& argv) {
    MonitorArgs out;

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") return {std::nullopt, "__help__"};

        auto fb = splitFlag(tok);
        const std::string& key = fb.key;
        const SubcommandOptionResult simple =
            parseMonitorSimpleOption(fb, argv, i, out);
        if (simple.found) {
            if (!simple.error.empty()) return {std::nullopt, simple.error};
            continue;
        }

        if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + key};
        } else {
            return {std::nullopt,
                    "unexpected argument: " + tok +
                    " (`gpufl monitor` does not launch a target process; use `gpufl trace -- <cmd>`)"};
        }
    }

    return {out, ""};
}

InfoParseResult parseInfoArgs(const std::vector<std::string>& argv) {
    InfoArgs out;

    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& tok = argv[i];
        if (tok == "-h" || tok == "--help") {
            return {std::nullopt, "__help__"};
        }
        auto fb = splitFlag(tok);
        const SubcommandOptionResult simple =
            parseInfoSimpleOption(fb, argv, i, out);
        if (simple.found) {
            if (!simple.error.empty()) return {std::nullopt, simple.error};
            continue;
        }
        if (!tok.empty() && tok[0] == '-') {
            return {std::nullopt, "unknown flag: " + fb.key};
        } else {
            return {std::nullopt,
                    "unexpected argument: " + tok +
                    " (`gpufl info` does not accept positional arguments)"};
        }
    }

    return {out, ""};
}

std::string validateTraceExecutionMode(const TraceArgs& args) {
    if (args.deep_requested && !args.passes.empty()) {
        return "--passes cannot be combined with --deep-* flags.\n"
               "\n"
               "  --passes runs the engines you name, relaunching the target "
               "once per pass.\n"
               "  --deep-* runs ONE adaptive pass: gpufl selects a compatible "
               "deep engine\n"
               "  and arms it only inside the window.\n"
               "\n"
               "Drop --passes to use a deep window.";
    }
    return validateTraceSegmentation(args);
}

bool segmentationRequested(const TraceArgs& args) {
    return args.segment_every_ms > 0 || args.segment_max_rows > 0 ||
           args.run_roll_every_ms > 0 || args.run_roll_max_bytes > 0;
}

std::string segmentationWarning(const TraceArgs& args) {
    if (args.run_roll_every_ms > 0 && args.segment_every_ms > 0 &&
        args.segment_every_ms > args.run_roll_every_ms / 10) {
        return "--segment-every is more than a tenth of --roll-every, so a run "
               "part can overshoot its budget visibly; a run part ends only at "
               "a segment boundary";
        }
    return {};
}

std::string validateTraceSegmentation(
    const TraceArgs& args,
    const std::string& inherited_analysis_id) {
    if (args.segment_every_ms < 0) {
        return "--segment-every cannot be negative";
    }
    if (args.segment_every_ms > 0 &&
        args.segment_every_ms < kMinSegmentEveryMs) {
        return "--segment-every must be at least 60s; shorter cadences can "
               "create a session storm";
    }

    if (args.run_roll_every_ms < 0) {
        return "--roll-every cannot be negative";
    }

    if (args.run_roll_every_ms > 0 && args.segment_every_ms <= 0) {
        return "--roll-every requires --segment-every. A run part ends at the "
               "next segment boundary, and with no segment time trigger a "
               "quiet period produces no boundary, so the part would grow "
               "without bound";
    }
    if (args.run_roll_every_ms > 0 &&
        args.run_roll_every_ms < args.segment_every_ms) {
        return "--roll-every must be at least --segment-every; a run part "
               "cannot be shorter than the segment carrying its boundary";
        }
    if (args.run_roll_max_bytes > 0 && args.segment_every_ms <= 0 &&
        args.segment_max_rows == 0) {
        return "--roll-max-bytes requires --segment-every or "
               "--segment-max-rows; at least one segment trigger must be armed "
               "for a run part to have a boundary to end on";
    }

    if (!segmentationRequested(args)) return {};

    if (!inherited_analysis_id.empty()) {
        return "session segmentation cannot be combined with an inherited "
               "GPUFL_ANALYSIS_ID; unset GPUFL_ANALYSIS_ID before launching "
               "the target";
    }
    if (args.passes.size() > 1) {
        return "session segmentation cannot be combined with a multi-pass "
               "--passes list; segmented runs concatenate time while analysis "
               "passes overlay the same interval";
    }
    if (!args.passes.empty()) {
        const std::string& pass = args.passes.front();
        if (pass != "Trace" && pass != "PmSampling") {
            return "session segmentation V1 supports only a single Trace or "
                   "PmSampling pass. Unsupported pass: " + pass;
        }
    }

    // No explicit pass plus --deep-* is the one supported composite: the
    // launcher pins native Trace as the base and prepares window-only PM.
    // No explicit pass and no deep flags is the ordinary single Trace pass.
    return {};
}

std::string generateRunId() {
    std::array<uint8_t, 16> bytes{};
    static thread_local std::mt19937_64 rng([] {
        std::random_device rd;
        std::seed_seq seed{
            rd(), rd(), rd(), rd(),
            static_cast<unsigned>(
                std::chrono::steady_clock::now().time_since_epoch().count())};
        return std::mt19937_64(seed);
    }());
    for (size_t i = 0; i < bytes.size(); i += sizeof(uint64_t)) {
        const uint64_t word = rng();
        for (size_t j = 0; j < sizeof(uint64_t); ++j) {
            bytes[i + j] = static_cast<uint8_t>(word >> (j * 8));
        }
    }

    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3f) | 0x80);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return out.str();
}

bool segmentationRuntimeReady() {
    return gpufl::segmentation::kRuntimeReady;
}

CaptureMode resolveCaptureMode(const TraceArgs& args) {
    return args.deep_requested ? CaptureMode::AdaptiveDeepWindow
                               : CaptureMode::ExplicitPasses;
}

AdaptiveCapturePlan resolveAdaptivePlan(const TraceArgs& args) {
    AdaptiveCapturePlan plan;
    if (!args.deep_requested) return plan;   // selected_deep stays empty

    // Trace is pinned as the base rather than left to the deep engine's own
    // policy: kernel_launch_rate and recent_kernel_ms are computed from its
    // completed-kernel records, and a rule that silently loses its metric
    // reads as "condition never held".
    plan.base = "Trace";

    // PM only, for now. It is the one deep engine that works on a 3090 at all:
    // PC sampling fails configuration under injection there and SASS emits no
    // records, so neither has a dormant cost anyone has measured.
    //
    // Earlier overhead measurements covered PM selected-but-not-initialized,
    // because initialization still lived in the arm path at the time. They are
    // intentionally not used as a policy budget here; the prepared-and-idle
    // configuration must be measured again after the lifecycle split.
    plan.selected_deep = {"PmSampling"};
    plan.arm_window_only = true;
    return plan;
}

std::vector<std::string> resolvePassPlan(const TraceArgs& args) {
    if (!args.passes.empty()) return args.passes;

    // Exactly one pass for an adaptive run. Relaunching the target per pass is
    // what --passes is for; a window that triggers on a live condition cannot
    // be reproduced across relaunches, so splitting it would change what is
    // being measured.
    if (resolveCaptureMode(args) == CaptureMode::AdaptiveDeepWindow) {
        const AdaptiveCapturePlan plan = resolveAdaptivePlan(args);
        std::string composite = plan.base;
        for (const std::string& engine : plan.selected_deep) {
            composite += "+" + engine;
        }
        return {composite};
    }
    return {"Trace"};
}

}  // namespace gpufl::launcher
