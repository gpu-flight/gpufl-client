#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace gpufl {

struct WindowTiming {
    /** Session/process monotonic clock. Valid only within this run. */
    std::int64_t opened_mono_ms = -1;
    std::int64_t closed_mono_ms = -1;
};

struct WindowMetadata {
    std::string window_id;
    std::string session_id;
    std::string channel;
    std::size_t window_sequence = 0;
    WindowTiming timing;
    std::int64_t created_wall_ms = 0;
    std::string payload_file;
    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_crc32 = 0;
};

/**
 * Immutable sidecar and post-ACK tombstone for one transport window.
 *
 * The payload may later be deleted, but this small file remains until the
 * session is retired. That keeps the sequence consumed and gives the agent a
 * stable idempotency key independent of filename timestamps.
 */
bool ensureWindowMetadata(const std::filesystem::path& session_dir,
                          const std::string& session_id,
                          const std::string& channel,
                          std::size_t sequence,
                          const std::filesystem::path& payload,
                          const WindowTiming& timing = {});

std::filesystem::path windowMetadataPath(
    const std::filesystem::path& session_dir,
    const std::string& channel,
    std::size_t sequence);

/** Include metadata tombstones when restoring the next channel sequence. */
void scanWindowMetadataMaxSequence(const std::filesystem::path& session_dir,
                                   const std::string& channel,
                                   std::size_t& max_sequence);

}  // namespace gpufl
