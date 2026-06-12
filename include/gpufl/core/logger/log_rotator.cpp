#include "gpufl/core/logger/log_rotator.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/log_salvage.hpp"

namespace gpufl {
namespace fs = std::filesystem;

LogFileRotator::LogFileRotator(LogRotationOptions opt, IFileCompressor* compressor)
    : opt_(std::move(opt)), compressor_(compressor) {}

std::string LogFileRotator::sessionDir() const {
    // base_path is treated as a directory in v1.2+. Strip a legacy
    // ".log" suffix so a caller passing the old "<dir>/<prefix>.log"
    // shape still gets a sensible parent dir to nest the session
    // subdir under (we just drop the ".log" and use what's left as
    // the directory path).
    fs::path p(opt_.base_path);
    if (p.extension() == ".log") {
        p.replace_extension();
    }
    p /= opt_.session_id;
    return p.string();
}

std::string LogFileRotator::tempDir() const { return sessionDir() + "/.tmp"; }

std::string LogFileRotator::activePath() const {
    std::ostringstream oss;
    oss << tempDir() << "/" << opt_.channel_name << ".log";
    return oss.str();
}

std::string LogFileRotator::rotatedPath(std::size_t index) const {
    std::ostringstream oss;
    oss << sessionDir() << "/" << opt_.channel_name << "." << index << ".log";
    return oss.str();
}

LogFileRotator::ExportWindowResult LogFileRotator::exportWindow_() const {
    const std::string active = activePath();
    std::error_code ec;
    if (!fs::exists(active, ec)) return ExportWindowResult::NoData;
    if (fs::file_size(active, ec) == 0) return ExportWindowResult::NoData;

    // Append-style monotonic index (higher = newer). Published files and
    // unpublished .tmp staging both count, so a failed publish cannot be
    // overwritten by the next rotation.
    const std::size_t next =
        nextLogWindowIndex(fs::path(sessionDir()), opt_.channel_name);

    if (!compressor_) {
        const std::string target = rotatedPath(next);
        fs::rename(active, target, ec);
        if (ec) {
            GFL_LOG_ERROR("[Logger] window export: publish failed for '",
                          active, "' (", ec.message(),
                          ") - deferred in the active file.");
            return ExportWindowResult::DeferredInActive;
        }
        pruneLogWindows(fs::path(sessionDir()), opt_.channel_name,
                        opt_.max_files);
        return ExportWindowResult::Published;
    }

    const std::string target = rotatedPath(next) + ".gz";
    std::ostringstream stg;
    stg << tempDir() << "/" << opt_.channel_name << "." << next << ".log.gz";
    const std::string staging = stg.str();

    // 1. gzip the active file into staging (inside .tmp) - a pure READ of
    // the active file, immune to holders. The session root never sees a
    // partial file.
    if (!compressor_->compressTo(active, staging)) {
        std::error_code rm_ec;
        fs::remove(staging, rm_ec);
        GFL_LOG_ERROR("[Logger] window export: compress failed for '", active,
                      "' - deferred to the next write.");
        return ExportWindowResult::DeferredInActive;
    }

    // 2. Truncate the active file BEFORE publishing the export: if this
    // is denied (holder without write sharing - rare), drop staging and
    // defer. The data then exists exactly once, in the active file.
    {
        std::ofstream trunc(active, std::ios::out | std::ios::trunc);
        if (!trunc) {
            std::error_code rm_ec;
            fs::remove(staging, rm_ec);
            GFL_LOG_ERROR("[Logger] window export: truncate denied for '",
                          active, "' - deferred to the next write.");
            return ExportWindowResult::DeferredInActive;
        }
    }

    // 3. Publish: staging → <channel>.<N>.log.gz in the session root.
    // Consumers only ever see the finished file. If the rename is blocked
    // (AV grabbing brand-new files), the data survives in staging and the
    // launcher's salvage pass publishes it later.
    bool published = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        fs::rename(staging, target, ec);
        if (!ec) {
            published = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100 << attempt));
    }
    if (!published) {
        GFL_LOG_ERROR("[Logger] window export: publish failed for '", staging,
                      "' (", ec.message(),
                      ") - left for the salvage pass.");
        return ExportWindowResult::StagedForSalvage;
    }

    pruneLogWindows(fs::path(sessionDir()), opt_.channel_name, opt_.max_files);
    return ExportWindowResult::Published;
}

void LogFileRotator::rotate() const { exportWindow_(); }

void LogFileRotator::compressActive() const {
    const ExportWindowResult result = exportWindow_();
    // Best-effort removal of this channel's (now exported) active file.
    // If exportWindow_ deferred in the active file, leave it for the salvage
    // path instead of deleting the only copy of the window.
    if (result == ExportWindowResult::DeferredInActive) return;
    // The shared .tmp dir itself is removed once by FileLogSink::close()
    // after EVERY channel has closed - other channels' actives are still
    // open while the first one finalizes.
    std::error_code ec;
    fs::remove(activePath(), ec);
}


}  // namespace gpufl
