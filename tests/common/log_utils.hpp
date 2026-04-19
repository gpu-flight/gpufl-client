#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "gpufl/core/json/json.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl::test {

using JsonValue = ::gpufl::json::JsonValue;

/**
 * Create a unique temporary directory for a test's log files. Caller is
 * responsible for cleanup via `std::filesystem::remove_all(path)`.
 * Throws `std::runtime_error` on failure (should never fail on a sane system).
 */
std::filesystem::path MakeTempLogDir();

/**
 * NDJSON contents of the three log channels the runtime emits.
 * Missing files yield empty vectors, not an error.
 */
struct LogEvents {
    std::vector<JsonValue> device;
    std::vector<JsonValue> scope;
    std::vector<JsonValue> system;
};

/**
 * Read all `{prefix}.{channel}.log` files (including rotated `.N.log`
 * variants) from `dir`. Channels read: "device", "scope", "system".
 */
LogEvents ReadAllLogs(const std::filesystem::path& dir,
                      const std::string& prefix);

/** Return events whose `"type"` field equals `type`. */
std::vector<JsonValue> FilterByType(const std::vector<JsonValue>& events,
                                    const std::string& type);

/**
 * Scan `profile_sample_batch` events and count the total number of rows
 * whose `sample_kind` column equals `kind`. Robust against the batch format
 * (one event can carry many rows).
 */
int CountProfileSamplesOfKind(const std::vector<JsonValue>& device_events,
                              const std::string& kind);

/**
 * Return the array of string values from the `field` field of the first
 * event matching `type`. Empty vector if no such event or field.
 */
std::vector<std::string> GetStringArrayField(
    const std::vector<JsonValue>& events, const std::string& type,
    const std::string& field);

/** Human-readable, filesystem-safe name for gtest parameterization. */
const char* EngineName(gpufl::ProfilingEngine e);

}  // namespace gpufl::test
