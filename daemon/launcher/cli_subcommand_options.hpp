#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "cli_parse_internal.hpp"

namespace gpufl::launcher {

struct SubcommandOptionResult {
    bool found = false;
    std::string error;
};

SubcommandOptionResult parseUploadSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, UploadArgs& args);
SubcommandOptionResult parseMonitorSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, MonitorArgs& args);
SubcommandOptionResult parseInfoSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, InfoArgs& args);

}  // namespace gpufl::launcher
