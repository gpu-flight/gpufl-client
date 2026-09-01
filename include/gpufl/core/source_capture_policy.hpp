#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpufl {

/**
 * Hard client-side bounds applied before correlated source is retained.
 *
 * These are capture limits, not backend request limits. Keeping them here
 * prevents a profiler-reported path from causing an unbounded read in the
 * target process.
 */
struct SourceCaptureLimits {
    std::size_t max_files = 64;
    std::uint64_t max_bytes_per_file = 1024 * 1024;
    std::uint64_t max_total_bytes = 8 * 1024 * 1024;
    std::size_t max_line_bytes = 64 * 1024;
    std::size_t max_manifest_entries = 256;
};

/**
 * Public source-capture policy carried by InitOptions and MonitorOptions.
 *
 * Empty approved_roots is deliberately fail-closed inside the collector.
 * Normal startup fills it with the launcher's --source-root or the target's
 * current working directory. Embedded callers can supply more than one root.
 */
struct SourceCaptureSettings {
    std::vector<std::string> approved_roots;
    SourceCaptureLimits limits;
};

namespace detail {

enum class SourceCaptureDisposition {
    Captured,
    Disabled,
    InvalidPath,
    NoApprovedRoot,
    OutsideApprovedRoots,
    SymlinkEscape,
    ExcludedSystemRoot,
    UnsupportedExtension,
    NonTextContent,
    NotRegularFile,
    FileLimitExceeded,
    FileTooLarge,
    TotalBudgetExceeded,
    LineTooLong,
    ReadFailed,
    ChangedDuringRead,
};

const char* sourceCaptureDispositionName(SourceCaptureDisposition value);

struct SourceCaptureRecord {
    std::uint32_t source_file_id = 0;
    std::string logical_path;
    std::string discovery_reason;
    SourceCaptureDisposition disposition =
        SourceCaptureDisposition::InvalidPath;
    std::uint64_t bytes = 0;
};

struct SourceCaptureResult {
    SourceCaptureRecord record;
    std::vector<std::string> lines;
};

struct SourceCaptureManifest {
    bool enabled = false;
    std::size_t approved_root_count = 0;
    SourceCaptureLimits limits;
    std::vector<SourceCaptureRecord> files;
    std::uint64_t captured_files = 0;
    std::uint64_t captured_bytes = 0;
    std::uint64_t skipped_files = 0;
    std::uint64_t omitted_manifest_entries = 0;
};

/**
 * Stateful admission and bounded-read engine for discovered source paths.
 *
 * DictionaryManager serializes calls under its own mutex. This class does not
 * lock internally and must not be shared without an owning lock.
 */
class SourceCapturePolicy {
   public:
    void configure(bool enabled, const SourceCaptureSettings& settings);
    void reset();

    SourceCaptureResult capture(const std::string& discovered_path,
                                std::uint32_t source_file_id,
                                const std::string& discovery_reason);

    const SourceCaptureManifest& manifest() const { return manifest_; }
    bool manifestDirty() const { return manifest_dirty_; }
    void markManifestFlushed() { manifest_dirty_ = false; }

   private:
    std::string unavailableLogicalPath(
        const std::filesystem::path& discovered_path,
        std::uint32_t source_file_id) const;
    void record(SourceCaptureRecord value);

    bool enabled_ = false;
    SourceCaptureSettings settings_;
    std::vector<std::filesystem::path> approved_lexical_roots_;
    std::vector<std::filesystem::path> approved_roots_;
    SourceCaptureManifest manifest_;
    bool manifest_dirty_ = false;
};

}  // namespace detail
}  // namespace gpufl
