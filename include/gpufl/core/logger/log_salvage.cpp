#include "gpufl/core/logger/log_salvage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>

#include <zlib.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <unistd.h>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)
#endif
#endif
#endif

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/session_ownership.hpp"
#include "gpufl/core/logger/window_metadata.hpp"

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

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(),
                         suffix) == 0;
}

}  // namespace

MoveFileNoReplaceResult moveFileNoReplace(const fs::path& from,
                                          const fs::path& to,
                                          std::error_code& ec) {
    ec.clear();
#if defined(_WIN32)
    // MoveFileEx without MOVEFILE_REPLACE_EXISTING is the Windows atomic
    // no-clobber primitive. Do not request WRITE_THROUGH here: the cutover
    // path runs on the collector/writer and is intentionally metadata-only.
    if (::MoveFileExW(from.c_str(), to.c_str(), 0)) {
        return MoveFileNoReplaceResult::Moved;
    }
    const DWORD error = ::GetLastError();
    ec = std::error_code(static_cast<int>(error), std::system_category());
    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
        return MoveFileNoReplaceResult::DestinationExists;
    }
    return MoveFileNoReplaceResult::Failed;
#else
#if defined(__linux__) && defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(),
                  RENAME_NOREPLACE) == 0) {
        return MoveFileNoReplaceResult::Moved;
    }
    const int rename_error = errno;
    if (rename_error == EEXIST) {
        ec = std::error_code(rename_error, std::generic_category());
        return MoveFileNoReplaceResult::DestinationExists;
    }
    // Older kernels/filesystems may not implement renameat2. Fall back to an
    // atomic destination claim with link(2); all spool moves stay on the same
    // session filesystem.
    if (rename_error != ENOSYS && rename_error != EINVAL &&
        rename_error != EOPNOTSUPP) {
        ec = std::error_code(rename_error, std::generic_category());
        return MoveFileNoReplaceResult::Failed;
    }
#endif
    if (::link(from.c_str(), to.c_str()) != 0) {
        const int link_error = errno;
        ec = std::error_code(link_error, std::generic_category());
        return link_error == EEXIST
                   ? MoveFileNoReplaceResult::DestinationExists
                   : MoveFileNoReplaceResult::Failed;
    }
    if (::unlink(from.c_str()) == 0) {
        return MoveFileNoReplaceResult::Moved;
    }
    // The destination is already a hard link to the same complete bytes. Keep
    // both names visible for salvage rather than pretending the move finished.
    ec = std::error_code(errno, std::generic_category());
    return MoveFileNoReplaceResult::Failed;
#endif
}

bool isValidGzipFile(const fs::path& path) {
    // A zero-length file is not a window, and zlib will not say so: gzread
    // reports EOF-on-first-read as a clean read with Z_OK, so an empty file
    // would validate. That mattered: a rename publishes a directory entry
    // before the data is necessarily durable, so a power loss right after
    // `.part` -> `.gz` can leave an empty `.gz` next to its still-complete
    // raw source. Validating it would delete the raw as a duplicate and
    // publish the empty file as the window - silent loss of a full window.
    // (A NON-empty truncated gzip is caught by the decode below.)
    std::error_code size_ec;
    const auto size = fs::file_size(path, size_ec);
    if (size_ec || size == 0) return false;

    gzFile file = gzopen(path.string().c_str(), "rb");
    if (!file) return false;

    bool ok = true;
    char buffer[64 * 1024];
    int read = 0;
    while ((read = gzread(file, buffer, sizeof(buffer))) > 0) {
    }
    if (read < 0) ok = false;

    int zerr = Z_OK;
    (void)gzerror(file, &zerr);
    if (zerr != Z_OK && zerr != Z_STREAM_END) ok = false;
    if (gzclose(file) != Z_OK) ok = false;
    return ok;
}

namespace {

constexpr const char* kTransportLossPrefix = ".gpufl-transport-loss.";
constexpr const char* kTransportLossSuffix = ".json";

bool isTransportLossMarker(const fs::path& path) {
    const std::string name = path.filename().string();
    return name.size() >
               std::strlen(kTransportLossPrefix) +
                   std::strlen(kTransportLossSuffix) &&
           name.compare(0, std::strlen(kTransportLossPrefix),
                        kTransportLossPrefix) == 0 &&
           name.compare(name.size() - std::strlen(kTransportLossSuffix),
                        std::strlen(kTransportLossSuffix),
                        kTransportLossSuffix) == 0;
}

std::string markerSafe(std::string value) {
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') c = '_';
    }
    return value.empty() ? "unknown" : value;
}

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char c : value) {
        switch (c) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return escaped;
}

bool recordTransportLossImpl(const fs::path& session_dir,
                             const std::string& channel,
                             const std::size_t index,
                             const std::string& reason) {
    const fs::path marker =
        session_dir /
        (std::string(kTransportLossPrefix) + markerSafe(channel) + "." +
         std::to_string(index) + kTransportLossSuffix);
    std::error_code state_ec;
    if (fs::exists(marker, state_ec)) return true;

    static std::atomic<std::uint64_t> nonce{0};
    const auto tick = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
    const fs::path partial =
        fs::path(marker.string() + ".part." + std::to_string(tick) + "." +
                 std::to_string(nonce.fetch_add(1, std::memory_order_relaxed)));
    {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << "{\"schema_version\":1,\"type\":\"transport_window_loss\","
               "\"channel\":\""
            << jsonEscape(channel) << "\",\"window_index\":" << index
            << ",\"reason\":\"" << jsonEscape(reason) << "\"}\n";
        out.flush();
        if (!out) {
            out.close();
            std::error_code rm_ec;
            fs::remove(partial, rm_ec);
            return false;
        }
    }

    std::error_code move_ec;
    const auto moved = moveFileNoReplace(partial, marker, move_ec);
    if (moved == MoveFileNoReplaceResult::Moved ||
        moved == MoveFileNoReplaceResult::DestinationExists) {
        std::error_code rm_ec;
        fs::remove(partial, rm_ec);
        return true;
    }
    std::error_code rm_ec;
    fs::remove(partial, rm_ec);
    GFL_LOG_ERROR("[Logger] could not persist transport-loss marker '",
                  marker.string(), "' (", move_ec.message(),
                  "); preserving the damaged artifact instead.");
    return false;
}

bool gzipPayloadsEqual(const fs::path& lhs, const fs::path& rhs) {
    gzFile a = gzopen(lhs.string().c_str(), "rb");
    gzFile b = gzopen(rhs.string().c_str(), "rb");
    if (!a || !b) {
        if (a) (void)gzclose(a);
        if (b) (void)gzclose(b);
        return false;
    }

    std::array<unsigned char, 64 * 1024> a_buf{};
    std::array<unsigned char, 64 * 1024> b_buf{};
    bool same = true;
    for (;;) {
        const int a_read =
            gzread(a, a_buf.data(), static_cast<unsigned>(a_buf.size()));
        const int b_read =
            gzread(b, b_buf.data(), static_cast<unsigned>(b_buf.size()));
        if (a_read < 0 || b_read < 0 || a_read != b_read) {
            same = false;
            break;
        }
        if (a_read == 0) break;
        if (std::memcmp(a_buf.data(), b_buf.data(),
                        static_cast<std::size_t>(a_read)) != 0) {
            same = false;
            break;
        }
    }
    if (gzclose(a) != Z_OK) same = false;
    if (gzclose(b) != Z_OK) same = false;
    return same;
}

bool rawMatchesGzipPayload(const fs::path& raw, const fs::path& gzip) {
    std::ifstream in(raw, std::ios::binary);
    gzFile gz = gzopen(gzip.string().c_str(), "rb");
    if (!in || !gz) {
        if (gz) (void)gzclose(gz);
        return false;
    }

    std::array<char, 64 * 1024> raw_buf{};
    std::array<char, 64 * 1024> gz_buf{};
    bool same = true;
    for (;;) {
        in.read(raw_buf.data(), static_cast<std::streamsize>(raw_buf.size()));
        const auto raw_read = in.gcount();
        const int gz_read =
            gzread(gz, gz_buf.data(), static_cast<unsigned>(gz_buf.size()));
        if (gz_read < 0 || raw_read != gz_read) {
            same = false;
            break;
        }
        if (raw_read == 0) {
            if (in.bad()) same = false;
            break;
        }
        if (std::memcmp(raw_buf.data(), gz_buf.data(),
                        static_cast<std::size_t>(raw_read)) != 0) {
            same = false;
            break;
        }
    }
    if (gzclose(gz) != Z_OK) same = false;
    return same;
}

std::vector<fs::path> regularFiles(const fs::path& dir) {
    std::vector<fs::path> entries;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::error_code e_ec;
        if (entry.is_regular_file(e_ec)) entries.push_back(entry.path());
    }
    std::sort(entries.begin(), entries.end());
    return entries;
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

std::size_t transportLossMarkerCount(const fs::path& session_dir) {
    std::size_t count = 0;
    std::error_code ec;
    if (!fs::exists(session_dir, ec) || !fs::is_directory(session_dir, ec)) {
        return 0;
    }
    for (const auto& entry : fs::directory_iterator(session_dir, ec)) {
        std::error_code entry_ec;
        if (entry.is_regular_file(entry_ec) &&
            isTransportLossMarker(entry.path())) {
            ++count;
        }
    }
    return count;
}

bool recordTransportLossMarker(const fs::path& session_dir,
                               const std::string& channel,
                               const std::size_t index,
                               const std::string& reason) {
    return recordTransportLossImpl(session_dir, channel, index, reason);
}

std::size_t nextLogWindowIndex(const fs::path& session_dir,
                               const std::string& channel) {
    // Scan `.tmp` FIRST, the session root SECOND - the reverse of the
    // direction a window travels when it is published. A window renamed
    // `.tmp` -> root between the two scans is then counted TWICE (harmless)
    // instead of zero times (which would hand its index out again, and
    // fs::rename replaces its destination silently).
    //
    // Order alone is not sufficient protection, only cheap: the live
    // rotation path allocates from a per-channel counter (FileChannel), so
    // it never re-derives an index from a directory a worker is mutating.
    // This scan remains the seed for that counter and the allocator for the
    // single-threaded salvage/launcher paths.
    std::size_t max_index = 0;
    scanMaxIndex(session_dir / ".tmp", channel, max_index);
    scanMaxIndex(session_dir, channel, max_index);
    scanWindowMetadataMaxSequence(session_dir, channel, max_index);
    return max_index + 1;
}

LogSalvageResult salvageOwnedSessionTempDir(const fs::path& session_dir) {
    LogSalvageResult result;
    // A prior pass may already have discarded an unrecoverable artifact.
    // Count the durable marker even when `.tmp` is gone so an uploader that
    // starts later cannot mistake the now-clean directory for a complete
    // session.
    result.lost_windows =
        static_cast<int>(transportLossMarkerCount(session_dir));
    const fs::path tmp = session_dir / ".tmp";
    std::error_code ec;
    if (!fs::exists(tmp, ec) || !fs::is_directory(tmp, ec)) return result;

    GzipFileCompressor compressor;
    std::set<fs::path> skip;

    // A worker compresses to `.gz.part` and atomically renames it to `.gz`
    // only after gzclose succeeds. A crash can therefore leave raw + part;
    // the raw file is authoritative and the incomplete part is disposable.
    // A part with neither raw nor completed gzip has no trustworthy source,
    // so keep it visible and report deferred instead of guessing.
    for (const auto& path : regularFiles(tmp)) {
        const std::string name = path.filename().string();
        if (!endsWith(name, ".log.gz.part")) continue;

        const fs::path completed =
            fs::path(path.string().substr(0, path.string().size() - 5));
        const fs::path raw =
            fs::path(path.string().substr(0, path.string().size() - 8));
        std::error_code state_ec;
        if (fs::exists(raw, state_ec) || fs::exists(completed, state_ec)) {
            std::error_code rm_ec;
            if (!fs::remove(path, rm_ec) && fs::exists(path, state_ec)) {
                ++result.deferred;
                skip.insert(path);
            }
        } else {
            GFL_LOG_ERROR("[Logger] salvage found orphan partial gzip '",
                          path.string(),
                          "' with no raw or completed source; leaving it for "
                          "manual recovery.");
            ++result.deferred;
            skip.insert(path);
        }
    }

    auto entries = regularFiles(tmp);

    // Reconcile the only intentional two-file transition: a completed,
    // validated gzip may coexist briefly with its raw source while the worker
    // removes/truncates the raw. Prefer the gzip, but never publish it until
    // the raw is gone or empty. If the gzip is corrupt, discard it only when
    // the complete raw source still exists.
    for (const auto& path : entries) {
        if (skip.count(path) != 0) continue;
        std::string channel;
        std::size_t idx = 0;
        bool compressed = false;
        if (!parseWindowName(path.filename().string(), channel, idx,
                             compressed) ||
            !compressed) {
            continue;
        }

        const fs::path raw =
            fs::path(path.string().substr(0, path.string().size() - 3));
        std::error_code state_ec;
        const bool raw_exists =
            fs::exists(raw, state_ec) && fs::is_regular_file(raw, state_ec);
        // EXISTS is not RECOVERABLE. removeOrTruncateFile deliberately leaves
        // a zero-byte husk when it cannot unlink (a holder on Windows), so a
        // raw file can be present and carry nothing. Treating a husk as "the
        // complete raw source" would authorise deleting a corrupt-but-partly
        // readable gzip - the only remaining copy of that window.
        std::error_code raw_size_ec;
        const bool raw_recoverable =
            raw_exists && fs::file_size(raw, raw_size_ec) > 0 && !raw_size_ec;
        if (!isValidGzipFile(path)) {
            std::error_code size_ec;
            const auto gz_size = fs::file_size(path, size_ec);
            // An EMPTY artifact can never carry a window, so there is nothing
            // to preserve even when no raw source is left. Dropping it
            // matters: `.tmp` is the "session still writing" signal, and a
            // permanently deferred zero-byte file would pin it forever, so
            // the session would never look finished to the uploader or agent.
            const bool empty_artifact = !size_ec && gz_size == 0;
            if (raw_recoverable) {
                GFL_LOG_ERROR("[Logger] salvage refused corrupt/incomplete "
                              "gzip '", path.string(),
                              "' - recovering it from the raw source.");
            } else if (empty_artifact) {
                GFL_LOG_ERROR("[Logger] salvage discarded an EMPTY window "
                              "artifact '", path.string(),
                              "' with no recoverable raw source. The events "
                              "in that window are LOST.");
            } else {
                // Non-empty but undecodable, with no usable raw: the bytes
                // may still be partly recoverable by hand, so keep them and
                // let `.tmp` stay - a visibly unfinished session beats
                // deleting the last copy.
                GFL_LOG_ERROR("[Logger] salvage kept corrupt gzip '",
                              path.string(),
                              "' for manual recovery - its raw source is gone "
                              "or empty, so this is the only copy left.");
            }
            bool loss_recorded = true;
            if (empty_artifact && !raw_recoverable) {
                // Persist BEFORE deleting the last artifact. If the marker
                // cannot be made durable, leave the empty file deferred: an
                // unfinished session is preferable to invisible loss.
                loss_recorded = recordTransportLossImpl(
                    session_dir, channel, idx, "empty_gzip_no_raw");
            }
            if ((raw_recoverable || empty_artifact) && loss_recorded) {
                std::error_code rm_ec;
                fs::remove(path, rm_ec);
                if (!rm_ec || !fs::exists(path, state_ec)) {
                    continue;  // the recoverable raw (if any) is salvaged
                }
                if (raw_recoverable) skip.insert(raw);
            }
            ++result.deferred;
            skip.insert(path);
            continue;
        }

        if (raw_exists && !removeOrTruncateFile(raw.string())) {
            GFL_LOG_ERROR("[Logger] salvage has a complete gzip '",
                          path.string(), "' but could not remove/truncate its "
                          "duplicate raw source '", raw.string(), "'.");
            ++result.deferred;
            skip.insert(path);
            skip.insert(raw);
        }
    }

    bool staged_publish_blocked = false;
    for (const auto& path : entries) {
        if (skip.count(path) != 0) continue;
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
                if (isValidGzipFile(target)) {
                    // Same index does NOT imply same window: an allocator
                    // race can produce two different payloads. Only discard a
                    // byte-for-byte decoded duplicate. A distinct payload is
                    // preserved for explicit reindex/recovery.
                    if (gzipPayloadsEqual(path, target)) {
                        std::error_code rm_ec;
                        fs::remove(path, rm_ec);
                        if (rm_ec && fs::exists(path, e_ec)) {
                            ++result.deferred;
                        }
                    } else {
                        GFL_LOG_ERROR(
                            "[Logger] salvage found TWO DIFFERENT windows "
                            "claiming index ",
                            idx, " for channel '", channel,
                            "'. Preserving the staged copy '", path.string(),
                            "'; automatic deletion would lose data.");
                        ++result.deferred;
                        skip.insert(path);
                    }
                    continue;
                }
                GFL_LOG_ERROR("[Logger] salvage target already exists but is "
                              "not a valid gzip: '", target.string(), "'.");
                ++result.deferred;
                continue;
            }
            if (!ensureWindowMetadata(
                    session_dir, session_dir.filename().string(), channel,
                    idx, path)) {
                ++result.deferred;
                continue;
            }
            std::error_code mv_ec;
            const auto moved = moveFileNoReplace(path, target, mv_ec);
            if (moved != MoveFileNoReplaceResult::Moved) {
                if (moved == MoveFileNoReplaceResult::DestinationExists) {
                    GFL_LOG_ERROR(
                        "[Logger] salvage publish collision for '",
                        target.string(),
                        "'; preserving the staged window instead of "
                        "overwriting it.");
                }
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

        if (idx > 0) {
            const fs::path same_index_target =
                session_dir /
                (channel + "." + std::to_string(idx) + ".log.gz");
            if (fs::exists(same_index_target, e_ec)) {
                if (isValidGzipFile(same_index_target)) {
                    if (!rawMatchesGzipPayload(path, same_index_target)) {
                        GFL_LOG_ERROR(
                            "[Logger] salvage found a raw window and a "
                            "DIFFERENT published window at index ",
                            idx, " for channel '", channel,
                            "'. Preserving the raw copy for recovery.");
                        ++result.deferred;
                        skip.insert(path);
                    } else if (!removeOrTruncateFile(path.string())) {
                        ++result.deferred;
                    }
                    continue;
                }
                GFL_LOG_ERROR("[Logger] salvage found raw window '",
                              path.string(), "' but its published target is "
                              "corrupt: '", same_index_target.string(), "'.");
                ++result.deferred;
                continue;
            }
        } else {
            idx = nextLogWindowIndex(session_dir, channel);
        }
        const fs::path target =
            session_dir / (channel + "." + std::to_string(idx) + ".log.gz");
        const fs::path staging = path.string() + ".gz";
        const fs::path partial = staging.string() + ".part";
        if (!compressor.compressTo(path.string(), partial.string())) {
            ++result.deferred;
            std::error_code rm_ec;
            fs::remove(partial, rm_ec);
            continue;
        }
        std::error_code promote_ec;
        fs::rename(partial, staging, promote_ec);
        if (promote_ec) {
            ++result.deferred;
            std::error_code rm_ec;
            fs::remove(partial, rm_ec);
            continue;
        }
        if (!removeOrTruncateFile(path.string())) {
            ++result.deferred;
            continue;
        }
        if (!ensureWindowMetadata(
                session_dir, session_dir.filename().string(), channel, idx,
                staging)) {
            ++result.deferred;
            continue;
        }
        std::error_code publish_ec;
        const auto published =
            moveFileNoReplace(staging, target, publish_ec);
        if (published != MoveFileNoReplaceResult::Moved) {
            ++result.deferred;
            continue;
        }
        ++result.salvaged;
    }

    result.lost_windows =
        static_cast<int>(transportLossMarkerCount(session_dir));
    if (result.deferred == 0) {
        removeTempDirIfClean(tmp);
    }
    return result;
}

LogSalvageResult salvageSessionTempDir(const fs::path& session_dir) {
    std::string lock_error;
    auto ownership =
        SessionOwnershipLock::tryAcquire(session_dir, &lock_error);
    if (!ownership) {
        LogSalvageResult result;
        result.active_sessions_skipped = 1;
        result.lost_windows =
            static_cast<int>(transportLossMarkerCount(session_dir));
        GFL_LOG_DEBUG("[Logger] salvage skipped active session '",
                      session_dir.string(), "': ", lock_error);
        return result;
    }
    return salvageOwnedSessionTempDir(session_dir);
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
        total.lost_windows += r.lost_windows;
        total.active_sessions_skipped += r.active_sessions_skipped;
    }
    return total;
}

bool sessionTempDirHasDeferredData(const fs::path& session_dir) {
    return tempDirHasDeferredData(session_dir / ".tmp");
}

}  // namespace gpufl
