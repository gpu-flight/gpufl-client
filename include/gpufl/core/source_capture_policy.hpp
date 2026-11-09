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
    /**
     * Vendored code inside an approved root is not captured by default: a
     * file is skipped when a directory between the root and the file is
     * named like third_party, external, vendor, _deps, site-packages, .venv
     * or node_modules (case-insensitive), or lies under a git submodule path
     * listed in <root>/.gitmodules. True captures it like project code.
     */
    bool include_third_party = false;
    /**
     * Extra exclusions, applied even with include_third_party. A bare name
     * (no slash) matches any directory component below the root, like the
     * default names. A value with a slash is a directory relative to each
     * approved root, or an absolute path; its whole subtree is skipped.
     */
    std::vector<std::string> excluded_dirs;
};

/**
 * Separator joining excluded_dirs in GPUFL_SOURCE_EXCLUDE: the platform's
 * path-list separator, as in PATH. An entry cannot contain it.
 */
#ifdef _WIN32
constexpr char kSourceExcludeListSeparator = ';';
#else
constexpr char kSourceExcludeListSeparator = ':';
#endif

namespace detail {

enum class SourceCaptureDisposition {
    Captured,
    Disabled,
    InvalidPath,
    NoApprovedRoot,
    OutsideApprovedRoots,
    SymlinkEscape,
    ExcludedSystemRoot,
    ExcludedThirdParty,
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
    void configureThirdPartyExclusions();
    bool isThirdParty(const std::filesystem::path& canonical,
                      std::size_t canonical_root,
                      const std::filesystem::path& absolute) const;
    bool hasExcludedDirectoryName(const std::filesystem::path& file,
                                  const std::filesystem::path& root) const;

    bool enabled_ = false;
    SourceCaptureSettings settings_;
    std::vector<std::filesystem::path> approved_lexical_roots_;
    std::vector<std::filesystem::path> approved_roots_;
    // Lower-case directory names, and canonical directory subtrees, whose
    // files are ExcludedThirdParty. Built once per configure().
    std::vector<std::string> excluded_dir_names_;
    std::vector<std::filesystem::path> excluded_subtrees_;
    SourceCaptureManifest manifest_;
    bool manifest_dirty_ = false;
};

}  // namespace detail
}  // namespace gpufl
