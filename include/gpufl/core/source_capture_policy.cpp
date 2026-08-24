#include "gpufl/core/source_capture_policy.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <system_error>

namespace gpufl::detail {
namespace {

namespace fs = std::filesystem;

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

bool componentEquals(const fs::path& left, const fs::path& right) {
#ifdef _WIN32
    return lowerAscii(left.string()) == lowerAscii(right.string());
#else
    return left == right;
#endif
}

bool isWithin(const fs::path& candidate, const fs::path& root) {
    auto candidate_it = candidate.begin();
    for (auto root_it = root.begin(); root_it != root.end();
         ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end() ||
            !componentEquals(*candidate_it, *root_it)) {
            return false;
        }
    }
    return true;
}

std::vector<fs::path> existingCanonicalRoots(
    const std::vector<std::string>& roots) {
    std::vector<fs::path> result;
    for (const auto& value : roots) {
        if (value.empty()) continue;
        std::error_code ec;
        fs::path canonical = fs::weakly_canonical(fs::path(value), ec);
        if (ec || !fs::is_directory(canonical, ec) || ec) continue;
        if (std::none_of(result.begin(), result.end(),
                         [&](const fs::path& existing) {
                             return isWithin(canonical, existing) &&
                                    isWithin(existing, canonical);
                         })) {
            result.push_back(std::move(canonical));
        }
    }
    return result;
}

void appendEnvironmentRoot(std::vector<fs::path>& roots, const char* key) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return;
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(fs::path(raw), ec);
    if (!ec && fs::is_directory(canonical, ec) && !ec) {
        roots.push_back(std::move(canonical));
    }
}

void appendEnvironmentChildRoot(std::vector<fs::path>& roots,
                                const char* key,
                                const fs::path& child) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return;
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(fs::path(raw) / child, ec);
    if (!ec && fs::is_directory(canonical, ec) && !ec) {
        roots.push_back(std::move(canonical));
    }
}

std::vector<fs::path> excludedSystemRoots() {
    std::vector<fs::path> roots;
#ifdef _WIN32
    appendEnvironmentRoot(roots, "CUDA_PATH");
    appendEnvironmentChildRoot(roots, "ProgramFiles",
                               "NVIDIA GPU Computing Toolkit");
    appendEnvironmentChildRoot(roots, "ProgramFiles(x86)",
                               "NVIDIA GPU Computing Toolkit");
    appendEnvironmentRoot(roots, "SystemRoot");
#else
    const std::vector<std::string> defaults = {
        "/usr/include", "/usr/local/include", "/usr/local/cuda",
        "/opt/cuda", "/opt/rocm"};
    roots = existingCanonicalRoots(defaults);
    appendEnvironmentRoot(roots, "CUDA_PATH");
#endif
    return roots;
}

bool hasAllowedExtension(const fs::path& path) {
    static constexpr std::array<const char*, 13> extensions = {
        ".c",   ".cc",  ".cpp", ".cxx", ".cu",  ".cuh", ".h",
        ".hh",  ".hpp", ".hxx", ".inc", ".inl", ".tpp"};
    const std::string extension = lowerAscii(path.extension().string());
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](const char* allowed) {
                           return extension == allowed;
                       });
}

std::string logicalPathFor(const fs::path& canonical,
                           const fs::path& root,
                           const std::size_t root_index,
                           const std::size_t root_count) {
    std::error_code ec;
    fs::path relative = fs::relative(canonical, root, ec);
    if (ec || relative.empty()) relative = canonical.filename();
    const std::string logical = relative.generic_string();
    if (root_count <= 1) return logical;
    return "root-" + std::to_string(root_index) + "/" + logical;
}

bool splitLines(const std::string& content, const std::size_t max_line_bytes,
                std::vector<std::string>& lines) {
    std::size_t line_start = 0;
    while (line_start < content.size()) {
        const std::size_t newline = content.find('\n', line_start);
        const std::size_t line_end =
            newline == std::string::npos ? content.size() : newline;
        std::size_t length = line_end - line_start;
        if (length > 0 && content[line_end - 1] == '\r') --length;
        if (length > max_line_bytes) return false;
        lines.emplace_back(content.substr(line_start, length));
        if (newline == std::string::npos) break;
        line_start = newline + 1;
    }
    return true;
}

}  // namespace

const char* sourceCaptureDispositionName(
    const SourceCaptureDisposition value) {
    switch (value) {
        case SourceCaptureDisposition::Captured:
            return "captured";
        case SourceCaptureDisposition::Disabled:
            return "capture_disabled";
        case SourceCaptureDisposition::InvalidPath:
            return "invalid_path";
        case SourceCaptureDisposition::NoApprovedRoot:
            return "no_approved_root";
        case SourceCaptureDisposition::OutsideApprovedRoots:
            return "outside_approved_roots";
        case SourceCaptureDisposition::SymlinkEscape:
            return "symlink_escape";
        case SourceCaptureDisposition::ExcludedSystemRoot:
            return "excluded_system_root";
        case SourceCaptureDisposition::UnsupportedExtension:
            return "unsupported_extension";
        case SourceCaptureDisposition::NonTextContent:
            return "non_text_content";
        case SourceCaptureDisposition::NotRegularFile:
            return "not_regular_file";
        case SourceCaptureDisposition::FileLimitExceeded:
            return "file_limit_exceeded";
        case SourceCaptureDisposition::FileTooLarge:
            return "file_too_large";
        case SourceCaptureDisposition::TotalBudgetExceeded:
            return "total_budget_exceeded";
        case SourceCaptureDisposition::LineTooLong:
            return "line_too_long";
        case SourceCaptureDisposition::ReadFailed:
            return "read_failed";
        case SourceCaptureDisposition::ChangedDuringRead:
            return "changed_during_read";
    }
    return "invalid_path";
}

void SourceCapturePolicy::configure(
    const bool enabled, const SourceCaptureSettings& settings) {
    enabled_ = enabled;
    settings_ = settings;
    approved_roots_ = existingCanonicalRoots(settings.approved_roots);
    manifest_ = {};
    manifest_.enabled = enabled_;
    manifest_.approved_root_count = approved_roots_.size();
    manifest_.limits = settings_.limits;
    manifest_dirty_ = true;
}

void SourceCapturePolicy::reset() {
    enabled_ = false;
    settings_ = {};
    approved_roots_.clear();
    manifest_ = {};
    manifest_dirty_ = false;
}

std::string SourceCapturePolicy::unavailableLogicalPath(
    const fs::path& discovered_path,
    const std::uint32_t source_file_id) const {
    std::string filename = discovered_path.filename().generic_string();
    if (filename.empty()) filename = "unknown";
    return "unavailable/source-" + std::to_string(source_file_id) + "/" +
           filename;
}

void SourceCapturePolicy::record(SourceCaptureRecord value) {
    if (value.disposition == SourceCaptureDisposition::Captured) {
        ++manifest_.captured_files;
        manifest_.captured_bytes += value.bytes;
    } else {
        ++manifest_.skipped_files;
    }

    if (manifest_.files.size() < settings_.limits.max_manifest_entries) {
        manifest_.files.push_back(std::move(value));
    } else {
        ++manifest_.omitted_manifest_entries;
    }
    manifest_dirty_ = true;
}

SourceCaptureResult SourceCapturePolicy::capture(
    const std::string& discovered_path,
    const std::uint32_t source_file_id,
    const std::string& discovery_reason) {
    SourceCaptureResult result;
    result.record.source_file_id = source_file_id;
    result.record.discovery_reason = discovery_reason;
    const fs::path input(discovered_path);
    result.record.logical_path =
        unavailableLogicalPath(input, source_file_id);

    const auto reject = [&](const SourceCaptureDisposition disposition,
                            const std::uint64_t bytes = 0) {
        result.record.disposition = disposition;
        result.record.bytes = bytes;
        record(result.record);
        return result;
    };

    if (!enabled_) return reject(SourceCaptureDisposition::Disabled);
    if (discovered_path.empty()) {
        return reject(SourceCaptureDisposition::InvalidPath);
    }
    if (approved_roots_.empty()) {
        return reject(SourceCaptureDisposition::NoApprovedRoot);
    }

    std::error_code ec;
    const fs::path absolute = fs::absolute(input, ec).lexically_normal();
    if (ec) return reject(SourceCaptureDisposition::InvalidPath);
    const fs::path canonical = fs::weakly_canonical(absolute, ec);
    if (ec) return reject(SourceCaptureDisposition::InvalidPath);

    std::size_t lexical_root = approved_roots_.size();
    std::size_t canonical_root = approved_roots_.size();
    for (std::size_t i = 0; i < approved_roots_.size(); ++i) {
        if (lexical_root == approved_roots_.size() &&
            isWithin(absolute, approved_roots_[i])) {
            lexical_root = i;
        }
        if (canonical_root == approved_roots_.size() &&
            isWithin(canonical, approved_roots_[i])) {
            canonical_root = i;
        }
    }
    if (canonical_root == approved_roots_.size()) {
        return reject(lexical_root != approved_roots_.size()
                          ? SourceCaptureDisposition::SymlinkEscape
                          : SourceCaptureDisposition::OutsideApprovedRoots);
    }

    result.record.logical_path = logicalPathFor(
        canonical, approved_roots_[canonical_root], canonical_root,
        approved_roots_.size());

    for (const auto& excluded : excludedSystemRoots()) {
        if (isWithin(canonical, excluded)) {
            return reject(SourceCaptureDisposition::ExcludedSystemRoot);
        }
    }
    if (!hasAllowedExtension(canonical)) {
        return reject(SourceCaptureDisposition::UnsupportedExtension);
    }
    const fs::file_status status = fs::status(canonical, ec);
    if (ec || !fs::is_regular_file(status)) {
        return reject(SourceCaptureDisposition::NotRegularFile);
    }
    if (manifest_.captured_files >= settings_.limits.max_files) {
        return reject(SourceCaptureDisposition::FileLimitExceeded);
    }

    const std::uintmax_t size = fs::file_size(canonical, ec);
    if (ec || size > (std::numeric_limits<std::uint64_t>::max)()) {
        return reject(SourceCaptureDisposition::ReadFailed);
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(size);
    if (bytes > settings_.limits.max_bytes_per_file) {
        return reject(SourceCaptureDisposition::FileTooLarge, bytes);
    }
    if (manifest_.captured_bytes > settings_.limits.max_total_bytes ||
        bytes > settings_.limits.max_total_bytes - manifest_.captured_bytes) {
        return reject(SourceCaptureDisposition::TotalBudgetExceeded, bytes);
    }

    std::ifstream stream(canonical, std::ios::binary);
    if (!stream.is_open()) {
        return reject(SourceCaptureDisposition::ReadFailed, bytes);
    }
    std::string content;
    content.reserve(static_cast<std::size_t>(bytes));
    std::array<char, 8192> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count <= 0) break;
        const auto chunk = static_cast<std::uint64_t>(count);
        if (content.size() > settings_.limits.max_bytes_per_file ||
            chunk > settings_.limits.max_bytes_per_file - content.size()) {
            return reject(SourceCaptureDisposition::ChangedDuringRead,
                          content.size() + chunk);
        }
        content.append(buffer.data(), static_cast<std::size_t>(count));
    }
    if (stream.bad()) {
        return reject(SourceCaptureDisposition::ReadFailed,
                      content.size());
    }
    if (content.size() != bytes) {
        return reject(SourceCaptureDisposition::ChangedDuringRead,
                      content.size());
    }
    if (content.find('\0') != std::string::npos) {
        return reject(SourceCaptureDisposition::NonTextContent, bytes);
    }
    if (!splitLines(content, settings_.limits.max_line_bytes, result.lines)) {
        result.lines.clear();
        return reject(SourceCaptureDisposition::LineTooLong, bytes);
    }

    result.record.disposition = SourceCaptureDisposition::Captured;
    result.record.bytes = bytes;
    record(result.record);
    return result;
}

}  // namespace gpufl::detail
