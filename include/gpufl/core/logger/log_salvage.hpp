#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace gpufl {

struct LogSalvageResult {
    int salvaged = 0;
    int deferred = 0;
};

/**
 * Return the next append-style window index for `channel` in a session.
 * Both published root files and unpublished `.tmp` staging files count, so
 * a failed publish cannot be overwritten by the next rotation.
 */
std::size_t nextLogWindowIndex(const std::filesystem::path& session_dir,
                               const std::string& channel);

/** Remove oldest published windows once more than `max_files` exist. */
void pruneLogWindows(const std::filesystem::path& session_dir,
                     const std::string& channel,
                     std::size_t max_files);

/** Publish staged `.tmp/*.log.gz` files and export non-empty `.tmp/*.log`. */
LogSalvageResult salvageSessionTempDir(
    const std::filesystem::path& session_dir);

/** Apply salvageSessionTempDir() to each session directory under `root`. */
LogSalvageResult salvageSessionTempDirs(
    const std::filesystem::path& root);

/** True when a session `.tmp` directory still holds uploadable data. */
bool sessionTempDirHasDeferredData(
    const std::filesystem::path& session_dir);

}  // namespace gpufl
