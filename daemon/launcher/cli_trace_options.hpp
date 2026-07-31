#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "cli_parse_internal.hpp"

namespace gpufl::launcher {

// Help grouping for `gpufl trace` options. Nonzero because the option manager
// treats section 0 as "undocumented"; the order here is the order help prints.
enum class TraceHelpSection {
    Capture = 1,
    Runtime,
    Segmentation,
    Window,
    Deep,
    Sampling,
};

struct TraceSimpleOptionResult {
    bool found = false;
    std::string error;
};

// Parses one `gpufl trace` option from the registry. A false found value means
// the token is not an option at all, which parseTraceArgs() reports as either a
// missing `--` separator or an unknown flag.
TraceSimpleOptionResult parseTraceSimpleOption(
    const detail::FlagBreak& flag,
    const std::vector<std::string>& argv,
    std::size_t& index,
    TraceArgs& args);

// Renders the registered options of one section, in registration order.
std::string formatTraceSimpleOptions(TraceHelpSection section);

// Every alias the registry knows, documented or not. Exposed so a test can
// assert the registry and the rendered help do not drift apart.
std::vector<std::string> traceOptionAliases();

}  // namespace gpufl::launcher
