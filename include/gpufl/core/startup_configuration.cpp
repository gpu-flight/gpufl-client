#include "gpufl/core/startup_configuration.hpp"

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include "gpufl/core/config_file_loader.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/segmentation_config.hpp"
#include "gpufl/core/version.hpp"

namespace gpufl::detail {
namespace {

bool parseNonNegativeEnv(const char* key, uint64_t& value,
                         std::string& error) {
    value = 0;
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return true;
    if (*raw == '-') {
        error = std::string(key) + " must be a non-negative integer";
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        error = std::string(key) + "='" + raw +
                "' is invalid (expected a non-negative integer)";
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool isUuidV4(const char* value) {
    if (!value || std::strlen(value) != 36) return false;
    for (size_t i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(value[i]))) {
            return false;
        }
    }
    if (value[14] != '4') return false;
    const char variant =
        static_cast<char>(std::tolower(static_cast<unsigned char>(value[19])));
    return variant == '8' || variant == '9' || variant == 'a' ||
           variant == 'b';
}

}  // namespace

void resolveStartupOptions(InitOptions& options) {
    std::string config_path = options.config_file;
    if (config_path.empty()) {
        if (const char* value = std::getenv(env::kConfigFile)) {
            config_path = value;
        }
    }
    if (!config_path.empty()) {
        ConfigFileLoader::apply(options, config_path);
    }

    // The trace launcher owns this explicit source-capture decision and always
    // sets 1 or 0. Apply it after the config file so --no-source cannot be
    // silently undone by inherited target configuration. Embedded users that
    // do not set the variable retain their InitOptions behavior.
    if (const char* value = std::getenv(env::kIncludeSource)) {
        options.enable_source_collection = std::strcmp(value, "1") == 0;
    }
    if (const char* value = std::getenv(env::kSourceRoot); value && *value) {
        options.source_capture.approved_roots = {value};
    }
    if (options.enable_source_collection &&
        options.source_capture.approved_roots.empty()) {
        std::error_code ec;
        const std::filesystem::path current =
            std::filesystem::current_path(ec);
        if (!ec) {
            options.source_capture.approved_roots = {current.string()};
        }
    }

    std::string api_path = options.api_path;
    if (api_path.empty()) {
        if (const char* value = std::getenv(env::kApiPath)) {
            api_path = value;
        }
    }
    options.api_path = normalizeApiPath(api_path);
}

bool readStartupSegmentationOptions(StartupSegmentationOptions& options,
                                    std::string& error) {
    if (!parseNonNegativeEnv(env::kSegmentEveryMs, options.segment_every_ms,
                             error) ||
        !parseNonNegativeEnv(env::kSegmentMaxRows, options.segment_max_rows,
                             error) ||
        !parseNonNegativeEnv(env::kRunRollEveryMs,
                             options.run_roll_every_ms, error) ||
        !parseNonNegativeEnv(env::kRunRollMaxBytes,
                             options.run_roll_max_bytes, error)) {
        return false;
    }

    constexpr uint64_t kMaxSignedMilliseconds =
        static_cast<uint64_t>((std::numeric_limits<int64_t>::max)());
    if (options.segment_every_ms > kMaxSignedMilliseconds) {
        error = std::string(env::kSegmentEveryMs) +
                " exceeds the supported signed 64-bit millisecond range";
        return false;
    }
    if (options.run_roll_every_ms > kMaxSignedMilliseconds) {
        error = std::string(env::kRunRollEveryMs) +
                " exceeds the supported signed 64-bit millisecond range";
        return false;
    }

    if (!options.enabled()) return true;

    const char* run_id = std::getenv(env::kRunId);
    if (!run_id || !*run_id) {
        error = "Session segmentation requires GPUFL_RUN_ID. The launcher "
                "must generate one run ID before starting the target.";
        return false;
    }
    if (!isUuidV4(run_id)) {
        error = std::string(env::kRunId) + "='" + run_id +
                "' is invalid (expected a UUIDv4)";
        return false;
    }
    if (!segmentation::kRuntimeReady) {
        error = "This build does not include executable session segmentation.";
        return false;
    }

    options.run_id = run_id;
    return true;
}

}  // namespace gpufl::detail
