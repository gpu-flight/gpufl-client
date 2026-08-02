#include "gpufl/core/session_bootstrap.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <utility>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl::detail {
namespace {

std::string g_last_log_path;
std::string g_last_session_id;

std::string defaultLogPath(const std::string& app) {
    // log_path is a directory: sessions nest beneath it as
    // `<log_path>/<session_id>/<channel>.log`. Returning just the app keeps
    // this default aligned with the on-disk directory layout.
    return app;
}

void applyLoggingEnvironment(const InitOptions& options,
                             Logger::Options& logger_options) {
    logger_options.system_sample_rate_ms = options.system_sample_rate_ms;
    logger_options.flush_always = options.flush_logs_always;
    if (const char* value = std::getenv(env::kFlushLogsAlways)) {
        std::string flag(value);
        std::transform(flag.begin(), flag.end(), flag.begin(),
                       [](const unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (flag == "1" || flag == "true" || flag == "yes" || flag == "on") {
            logger_options.flush_always = true;
        }
    }
    if (const char* value = std::getenv(env::kLogRotateBytes)) {
        if (const auto bytes = std::strtoull(value, nullptr, 10); bytes > 0) {
            logger_options.rotate_bytes = static_cast<std::size_t>(bytes);
        }
    }
    if (const char* value = std::getenv(env::kLogRotateAfterMs)) {
        if (const auto ms = std::strtoll(value, nullptr, 10); ms > 0) {
            logger_options.rotate_after_ms = static_cast<std::int64_t>(ms);
        }
    }
    if (const char* value = std::getenv(env::kLogMaxSpoolBytes)) {
        logger_options.max_spool_bytes = std::strtoull(value, nullptr, 10);
    }
    if (const char* value = std::getenv(env::kLogMinFreeBytes)) {
        logger_options.min_free_bytes = std::strtoull(value, nullptr, 10);
    }
}

std::shared_ptr<const RunPartContext> makeInitialRunPart(
    const Runtime& runtime, const bool segmented) {
    if (!segmented ||
        (runtime.run_roll_every_ms <= 0 && runtime.run_roll_max_bytes == 0)) {
        return nullptr;
    }
    const auto opened_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return std::make_shared<const RunPartContext>(
        runtime.run_id, runtime.run_id, std::string(), 1u, opened_mono_ns, 0u);
}

}  // namespace

bool openInitialSessionLogging(Runtime& runtime, const InitOptions& options,
                               const bool segmented,
                               InitialSessionLoggingState& state) {
    state.log_path = options.log_path.empty()
        ? defaultLogPath(runtime.app_name)
        : options.log_path;
    state.options.base_path = state.log_path;
    // The uploader discovers sessions from this directory layout instead of
    // parsing a job_start event out of a legacy flat log file.
    state.options.session_id = runtime.session_id;
    applyLoggingEnvironment(options, state.options);

    // Preserve the existing post-shutdown reporting state even if opening the
    // logger itself fails.
    g_last_log_path = state.log_path;
    g_last_session_id = runtime.session_id;

    const auto initial_run_part = makeInitialRunPart(runtime, segmented);
    if (initial_run_part && runtime.run_roll_max_bytes > 0) {
        state.options.on_serialized_bytes =
            [part = initial_run_part](const uint64_t bytes) noexcept {
                part->addSerializedBytes(bytes);
            };
    }

    GFL_LOG_DEBUG("Opening log file: ", state.log_path);
    if (!runtime.logger->open(state.options)) {
        GFL_LOG_ERROR("Failed to open logger at: ", state.log_path);
        return false;
    }

    auto dictionary = segmented
        ? std::make_shared<SegmentDictionaryEmitter>()
        : nullptr;
    if (!runtime.publishSegmentContext(std::make_shared<SegmentContext>(
            runtime.run_id, runtime.session_id, runtime.segment_index,
            GetTimestampNs(), runtime.logger, std::move(dictionary),
            initial_run_part))) {
        GFL_LOG_ERROR("Failed to publish the initial segment context");
        runtime.logger->close();
        return false;
    }
    return true;
}

LastSessionReportSource lastSessionReportSource() {
    return {g_last_log_path, g_last_session_id};
}

}  // namespace gpufl::detail
