#include "gpufl/core/logger/log_salvage.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <system_error>
#include <vector>

#include "gpufl/core/logger/file_compressor.hpp"

namespace gpufl {
namespace fs = std::filesystem;
namespace {

bool parseWindowName(const std::string& filename,
                     std::string& channel,
                     std::size_t& index,
                     bool& compressed) {
    std::string rest = filename;
    compressed = false;
    if (rest.size() > 3 && rest.compare(rest.size() - 3, 3, ".gz") == 0) {
        compressed = true;
        rest.erase(rest.size() - 3);
    }
    if (rest.size() <= 4 || rest.compare(rest.size() - 4, 4, ".log") != 0) {
        return false;
    }
    rest.erase(rest.size() - 4);

    const auto dot = rest.find_last_of('.');
    if (dot == std::string::npos) {
        channel = rest;
        index = 0;
        return !channel.empty();
    }
    try {
        const auto parsed = std::stoull(rest.substr(dot + 1));
        if (parsed == 0) return false;
        channel = rest.substr(0, dot);
        index = static_cast<std::size_t>(parsed);
        return !channel.empty();
    } catch (...) {
        channel = rest;
        index = 0;
        return !channel.empty();
    }
}

void scanMaxIndex(const fs::path& dir,
                  const std::string& channel,
                  std::size_t& max_index) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::error_code e_ec;
        if (!entry.is_regular_file(e_ec)) continue;
        std::string ch;
        std::size_t idx = 0;
        bool compressed = false;
        if (!parseWindowName(entry.path().filename().string(), ch, idx,
                             compressed)) {
            continue;
        }
        if (ch == channel && idx > max_index) max_index = idx;
    }
}

std::vector<std::size_t> publishedWindowIndices(const fs::path& session_dir,
                                                const std::string& channel) {
    std::set<std::size_t> indices;
    std::error_code ec;
    if (!fs::exists(session_dir, ec) || !fs::is_directory(session_dir, ec)) {
        return {};
    }
    for (const auto& entry : fs::directory_iterator(session_dir, ec)) {
        std::error_code e_ec;
        if (!entry.is_regular_file(e_ec)) continue;
        std::string ch;
        std::size_t idx = 0;
        bool compressed = false;
        if (parseWindowName(entry.path().filename().string(), ch, idx,
                            compressed) &&
            ch == channel && idx > 0) {
            indices.insert(idx);
        }
    }
    return {indices.begin(), indices.end()};
}

bool removeOrTruncate(const fs::path& p) {
    std::error_code ec;
    fs::remove(p, ec);
    if (!ec || !fs::exists(p, ec)) return true;

    std::ofstream trunc(p, std::ios::out | std::ios::trunc);
    return static_cast<bool>(trunc);
}

bool tempDirHasDeferredData(const fs::path& tmp) {
    std::error_code ec;
    if (!fs::exists(tmp, ec) || !fs::is_directory(tmp, ec)) return false;

    for (const auto& entry : fs::directory_iterator(tmp, ec)) {
        std::error_code e_ec;
        if (!entry.is_regular_file(e_ec)) continue;
        const auto size = fs::file_size(entry.path(), e_ec);
        if (e_ec || size > 0) return true;

        std::error_code rm_ec;
        fs::remove(entry.path(), rm_ec);
    }
    return ec != std::error_code{};
}

void removeTempDirIfClean(const fs::path& tmp) {
    if (tempDirHasDeferredData(tmp)) return;
    std::error_code ec;
    fs::remove_all(tmp, ec);
}

}  // namespace

std::size_t nextLogWindowIndex(const fs::path& session_dir,
                               const std::string& channel) {
    std::size_t max_index = 0;
    scanMaxIndex(session_dir, channel, max_index);
    scanMaxIndex(session_dir / ".tmp", channel, max_index);
    return max_index + 1;
}

void pruneLogWindows(const fs::path& session_dir,
                     const std::string& channel,
                     const std::size_t max_files) {
    if (max_files == 0) return;
    auto indices = publishedWindowIndices(session_dir, channel);
    if (indices.size() <= max_files) return;
    const std::size_t remove_count = indices.size() - max_files;
    for (std::size_t i = 0; i < remove_count; ++i) {
        const auto idx = indices[i];
        const fs::path base =
            session_dir / (channel + "." + std::to_string(idx) + ".log");
        std::error_code ec;
        fs::remove(base, ec);
        fs::remove(base.string() + ".gz", ec);
    }
}

LogSalvageResult salvageSessionTempDir(const fs::path& session_dir) {
    LogSalvageResult result;
    const fs::path tmp = session_dir / ".tmp";
    std::error_code ec;
    if (!fs::exists(tmp, ec) || !fs::is_directory(tmp, ec)) return result;

    std::vector<fs::path> entries;
    for (const auto& entry : fs::directory_iterator(tmp, ec)) {
        std::error_code e_ec;
        if (entry.is_regular_file(e_ec)) entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end());

    GzipFileCompressor compressor;
    bool staged_publish_blocked = false;
    for (const auto& path : entries) {
        std::error_code e_ec;
        if (!fs::exists(path, e_ec) || !fs::is_regular_file(path, e_ec)) {
            continue;
        }

        std::string channel;
        std::size_t idx = 0;
        bool compressed = false;
        const std::string name = path.filename().string();
        if (!parseWindowName(name, channel, idx, compressed)) {
            ++result.deferred;
            continue;
        }

        if (compressed) {
            if (idx == 0) idx = nextLogWindowIndex(session_dir, channel);
            fs::path target =
                session_dir /
                (channel + "." + std::to_string(idx) + ".log.gz");
            if (fs::exists(target, e_ec)) {
                idx = nextLogWindowIndex(session_dir, channel);
                target = session_dir /
                         (channel + "." + std::to_string(idx) + ".log.gz");
            }
            std::error_code mv_ec;
            fs::rename(path, target, mv_ec);
            if (mv_ec) {
                ++result.deferred;
                staged_publish_blocked = true;
            } else {
                ++result.salvaged;
            }
            continue;
        }

        if (staged_publish_blocked) {
            ++result.deferred;
            continue;
        }

        std::error_code sz_ec;
        const auto size = fs::file_size(path, sz_ec);
        if (sz_ec) {
            ++result.deferred;
            continue;
        }
        if (size == 0) {
            std::error_code rm_ec;
            fs::remove(path, rm_ec);
            continue;
        }

        idx = nextLogWindowIndex(session_dir, channel);
        const fs::path target =
            session_dir / (channel + "." + std::to_string(idx) + ".log.gz");
        if (!compressor.compressTo(path.string(), target.string())) {
            ++result.deferred;
            std::error_code rm_ec;
            fs::remove(target, rm_ec);
            continue;
        }
        ++result.salvaged;
        if (!removeOrTruncate(path)) ++result.deferred;
    }

    if (result.deferred == 0) {
        removeTempDirIfClean(tmp);
    }
    return result;
}

LogSalvageResult salvageSessionTempDirs(const fs::path& root) {
    LogSalvageResult total;
    std::error_code ec;
    if (root.empty() || !fs::exists(root, ec) || !fs::is_directory(root, ec)) {
        return total;
    }
    for (const auto& session : fs::directory_iterator(root, ec)) {
        std::error_code s_ec;
        if (!session.is_directory(s_ec)) continue;
        const auto r = salvageSessionTempDir(session.path());
        total.salvaged += r.salvaged;
        total.deferred += r.deferred;
    }
    return total;
}

bool sessionTempDirHasDeferredData(const fs::path& session_dir) {
    return tempDirHasDeferredData(session_dir / ".tmp");
}

}  // namespace gpufl
