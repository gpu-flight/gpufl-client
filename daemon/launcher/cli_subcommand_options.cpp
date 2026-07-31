#include "cli_subcommand_options.hpp"

#include "cli_option_manager.hpp"

namespace gpufl::launcher {

namespace {

template <typename Args, std::string Args::*Slot>
std::string parseString(const detail::FlagBreak& flag,
                        const std::vector<std::string>& argv,
                        std::size_t& index,
                        Args& args) {
    return detail::takeFlagValue(flag, argv, index, args.*Slot);
}

template <typename Args, std::string Args::*Slot>
std::string parseNonEmptyString(const detail::FlagBreak& flag,
                                const std::vector<std::string>& argv,
                                std::size_t& index,
                                Args& args) {
    std::string& value = args.*Slot;
    if (const std::string error = detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) {
        return error;
    }
    return value.empty() ? flag.key + " cannot be empty" : std::string();
}

template <typename Args, bool Args::*Slot>
std::string setFlag(const detail::FlagBreak& flag,
                    const std::vector<std::string>&,
                    std::size_t&,
                    Args& args) {
    if (flag.inline_value) return "unknown flag: " + flag.key;
    args.*Slot = true;
    return {};
}

std::string parseUploadTimeout(const detail::FlagBreak& flag,
                               const std::vector<std::string>& argv,
                               std::size_t& index,
                               UploadArgs& args) {
    std::string value;
    if (const std::string error = detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) return error;
    if (!detail::parseNonNegativeInt(value, args.timeout_s)) {
        return "invalid --timeout value: " + value +
               " (expected a non-negative integer, seconds)";
    }
    return {};
}

std::string parseUploadRetries(const detail::FlagBreak& flag,
                               const std::vector<std::string>& argv,
                               std::size_t& index,
                               UploadArgs& args) {
    std::string value;
    if (const std::string error = detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) return error;
    if (!detail::parseNonNegativeInt(value, args.retries)) {
        return "invalid --retries value: " + value +
               " (expected a non-negative integer)";
    }
    return {};
}

std::string parseMonitorInterval(const detail::FlagBreak& flag,
                                 const std::vector<std::string>& argv,
                                 std::size_t& index,
                                 MonitorArgs& args) {
    std::string value;
    if (const std::string error = detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) return error;
    if (!detail::parsePositiveInt(value, args.interval_ms)) {
        return "invalid --interval value: " + value +
               " (expected a positive integer, milliseconds)";
    }
    return {};
}

std::string parseInfoDevice(const detail::FlagBreak& flag,
                            const std::vector<std::string>& argv,
                            std::size_t& index,
                            InfoArgs& args) {
    std::string value;
    if (const std::string error = detail::takeFlagValue(flag, argv, index, value);
        !error.empty()) return error;
    int device_id = 0;
    if (!detail::parseNonNegativeInt(value, device_id)) {
        return "invalid --device value: " + value +
               " (expected a non-negative integer)";
    }
    args.device_id = device_id;
    return {};
}

const detail::CliOptionManager<UploadArgs>& uploadOptions() {
    static const detail::CliOptionManager<UploadArgs> options = [] {
        detail::CliOptionManager<UploadArgs> registry;
        registry
            .add({"-q", "--quiet"}, &setFlag<UploadArgs, &UploadArgs::quiet>)
            .add({"--all-sessions"}, &setFlag<UploadArgs, &UploadArgs::all_sessions>)
            .add({"--force"}, &setFlag<UploadArgs, &UploadArgs::force>)
            .add({"--backend-url"}, &parseString<UploadArgs, &UploadArgs::backend_url>)
            .add({"--api-key"}, &parseString<UploadArgs, &UploadArgs::api_key>)
            .add({"--api-path"}, &parseString<UploadArgs, &UploadArgs::api_path>)
            .add({"--agent-jar"}, &parseString<UploadArgs, &UploadArgs::agent_jar>)
            .add({"--timeout"}, &parseUploadTimeout)
            .add({"--retries"}, &parseUploadRetries);
        return registry;
    }();
    return options;
}

const detail::CliOptionManager<MonitorArgs>& monitorOptions() {
    static const detail::CliOptionManager<MonitorArgs> options = [] {
        detail::CliOptionManager<MonitorArgs> registry;
        registry
            .add({"-v", "--verbose"}, &setFlag<MonitorArgs, &MonitorArgs::verbose>)
            .add({"-q", "--quiet"}, &setFlag<MonitorArgs, &MonitorArgs::quiet>)
            .add({"--upload"}, &setFlag<MonitorArgs, &MonitorArgs::upload>)
            .add({"-n", "--name"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::name>)
            .add({"-o", "--output"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::output_dir>)
            .add({"--interval"}, &parseMonitorInterval)
            .add({"--backend-url"}, &parseString<MonitorArgs, &MonitorArgs::backend_url>)
            .add({"--api-key"}, &parseString<MonitorArgs, &MonitorArgs::api_key>)
            .add({"--api-version"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::api_version>)
            .add({"--agent-jar"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::agent_jar>)
            .add({"--agent-cursor"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::agent_cursor>)
            .add({"--log-types"}, &parseNonEmptyString<MonitorArgs, &MonitorArgs::log_types>);
        return registry;
    }();
    return options;
}

const detail::CliOptionManager<InfoArgs>& infoOptions() {
    static const detail::CliOptionManager<InfoArgs> options = [] {
        detail::CliOptionManager<InfoArgs> registry;
        registry
            .add({"--json"}, &setFlag<InfoArgs, &InfoArgs::json>)
            .add({"--device"}, &parseInfoDevice);
        return registry;
    }();
    return options;
}

template <typename Args>
SubcommandOptionResult parse(const detail::CliOptionManager<Args>& options,
                             const detail::FlagBreak& flag,
                             const std::vector<std::string>& argv,
                             std::size_t& index,
                             Args& args) {
    std::string error;
    const bool found = options.parse(flag, argv, index, args, error);
    return {found, std::move(error)};
}

}  // namespace

SubcommandOptionResult parseUploadSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, UploadArgs& args) {
    return parse(uploadOptions(), flag, argv, index, args);
}

SubcommandOptionResult parseMonitorSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, MonitorArgs& args) {
    return parse(monitorOptions(), flag, argv, index, args);
}

SubcommandOptionResult parseInfoSimpleOption(
    const detail::FlagBreak& flag, const std::vector<std::string>& argv,
    std::size_t& index, InfoArgs& args) {
    return parse(infoOptions(), flag, argv, index, args);
}

}  // namespace gpufl::launcher
