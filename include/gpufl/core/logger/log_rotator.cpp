#include "gpufl/core/logger/log_rotator.hpp"

#include <chrono>
#include <filesystem>
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

std::string LogFileRotator::retiredPath(std::size_t index) const {
    std::ostringstream oss;
    oss << tempDir() << "/" << opt_.channel_name << "." << index << ".log";
    return oss.str();
}

std::size_t LogFileRotator::nextWindowIndex() const {
    return nextLogWindowIndex(fs::path(sessionDir()), opt_.channel_name);
}

bool LogFileRotator::publishWithRetry_(const std::string& from,
                                       const std::string& to) const {
    std::error_code ec;
    // The no-replace operation closes the exists()->rename() TOCTOU window:
    // a concurrent salvage/uploader can claim `to` after a pre-check. A
    // collision must leave `from` visible, never replace the older window.
    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto moved = moveFileNoReplace(from, to, ec);
        if (moved == MoveFileNoReplaceResult::Moved) return true;
        if (moved == MoveFileNoReplaceResult::DestinationExists) {
            GFL_LOG_ERROR("[Logger] window export: refusing to publish '",
                          from, "' over the existing window '", to,
                          "' - window indices must be unique. Left in `.tmp` "
                          "for the salvage pass.");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100 << attempt));
    }
    GFL_LOG_ERROR("[Logger] window export: publish failed for '", from, "' (",
                  ec.message(), ") - left for the salvage pass.");
    return false;
}

LogFileRotator::RetireResult LogFileRotator::retireActiveWindow(
    const std::size_t index) const {
    const std::string active = activePath();
    std::error_code ec;
    if (!fs::exists(active, ec)) return RetireResult::NoData;
    if (fs::file_size(active, ec) == 0) return RetireResult::NoData;

    // Same reasoning as publishWithRetry_: claim the retired name atomically.
    // A pre-check followed by rename still permits another actor to create
    // the destination between the two operations.
    const auto retired =
        moveFileNoReplace(active, retiredPath(index), ec);
    if (retired == MoveFileNoReplaceResult::DestinationExists) {
        GFL_LOG_ERROR("[Logger] window cutover: index ", index,
                      " is already taken by '", retiredPath(index),
                      "' - refusing to overwrite it.");
        return RetireResult::Blocked;
    }
    if (retired != MoveFileNoReplaceResult::Moved) {
        GFL_LOG_ERROR("[Logger] window cutover: could not retire '", active,
                      "' (", ec.message(),
                      ") - the data stays in the active window and the next "
                      "beat retries.");
        return RetireResult::Blocked;
    }
    return RetireResult::Retired;
}

LogFileRotator::ExportWindowResult LogFileRotator::exportRetiredWindow(
    const std::size_t index, std::size_t* pruned_windows) const {
    const std::string retired = retiredPath(index);
    std::error_code ec;
    if (!fs::exists(retired, ec)) return ExportWindowResult::NoData;

    const auto prune = [&]() {
        const std::size_t removed = pruneLogWindows(
            fs::path(sessionDir()), opt_.channel_name, opt_.max_files);
        if (removed > 0) {
            if (pruned_windows) *pruned_windows += removed;
            GFL_LOG_ERROR("[Logger] window cap (max_files=", opt_.max_files,
                          ") deleted ", removed, " old '", opt_.channel_name,
                          "' window(s) that may not have been uploaded yet. "
                          "Raise max_files or drain windows faster.");
        }
    };

    if (!compressor_) {
        if (!publishWithRetry_(retired, rotatedPath(index))) {
            // The retired file is still in `.tmp`, indexed and complete -
            // the salvage pass publishes it.
            return ExportWindowResult::StagedForSalvage;
        }
        prune();
        return ExportWindowResult::Published;
    }

    const std::string staging = retired + ".gz";
    const std::string partial = staging + ".part";
    // A name ending in `.part` is deliberately NOT a transport window.
    // `compressTo` closes and validates the gzip before the atomic rename
    // below makes it visible to salvage. A crash during compression leaves
    // the raw authoritative source plus an explicitly incomplete artifact.
    //
    // SCOPE: this is PROCESS-CRASH safe (SIGKILL, an unhandled fault, a
    // teardown that never runs shutdown) - at every point the window's bytes
    // exist in exactly one place salvage can name. It is NOT power-loss
    // durable: nothing here fsyncs the gzip or the directory, so a rename
    // can reach the disk before the data it names. Salvage is defensive
    // about that (it re-validates every gzip and rejects empty or truncated
    // ones) but the guarantee itself needs file sync -> rename -> directory
    // sync, which is not implemented.
    fs::remove(partial, ec);
    if (!compressor_->compressTo(retired, partial)) {
        std::error_code rm_ec;
        fs::remove(partial, rm_ec);
        GFL_LOG_ERROR("[Logger] window export: compress failed for '", retired,
                      "' - left in `.tmp` for the salvage pass.");
        return ExportWindowResult::DeferredInActive;
    }

    fs::rename(partial, staging, ec);
    if (ec) {
        std::error_code rm_ec;
        fs::remove(partial, rm_ec);
        GFL_LOG_ERROR("[Logger] window export: could not promote completed "
                      "gzip '", partial, "' to '", staging, "' (",
                      ec.message(), ") - raw source remains authoritative.");
        return ExportWindowResult::StagedForSalvage;
    }

    // The completed staging gzip is now authoritative only after the raw
    // source is gone or empty. Never publish both: shutdown salvage would
    // otherwise assign the leftover raw a new index and duplicate every row.
    if (!removeOrTruncateFile(retired)) {
        GFL_LOG_ERROR("[Logger] window export: completed gzip '", staging,
                      "' but could not remove or truncate raw source '",
                      retired, "' - left both in `.tmp` for exact-once "
                      "salvage reconciliation.");
        return ExportWindowResult::StagedForSalvage;
    }

    if (!publishWithRetry_(staging, rotatedPath(index) + ".gz")) {
        return ExportWindowResult::StagedForSalvage;
    }
    prune();
    return ExportWindowResult::Published;
}

LogFileRotator::ExportWindowResult LogFileRotator::exportWindow_(
    std::size_t* pruned_windows) const {
    return exportWindowAt_(nextWindowIndex(), pruned_windows);
}

LogFileRotator::ExportWindowResult LogFileRotator::exportWindowAt_(
    const std::size_t index, std::size_t* pruned_windows) const {
    switch (retireActiveWindow(index)) {
        case RetireResult::NoData:
            return ExportWindowResult::NoData;
        case RetireResult::Blocked:
            return ExportWindowResult::DeferredInActive;
        case RetireResult::Retired:
            // Shutdown uses the same crash-safe raw -> .part -> completed
            // gzip transaction as mid-run retirement; only the thread differs.
            return exportRetiredWindow(index, pruned_windows);
    }
    return ExportWindowResult::DeferredInActive;
}

LogFileRotator::ExportWindowResult LogFileRotator::rotate(
    std::size_t* pruned_windows) const {
    return exportWindow_(pruned_windows);
}

LogFileRotator::ExportWindowResult LogFileRotator::compressActive(
    const std::size_t index) const {
    const ExportWindowResult result = exportWindowAt_(index, nullptr);
    // Best-effort removal of this channel's (now exported) active file.
    // If exportWindow_ deferred in the active file, leave it for the salvage
    // path instead of deleting the only copy of the window.
    if (result == ExportWindowResult::DeferredInActive) return result;
    // The shared .tmp dir itself is removed once by FileLogSink::close()
    // after EVERY channel has closed - other channels' actives are still
    // open while the first one finalizes.
    std::error_code ec;
    fs::remove(activePath(), ec);
    return result;
}


}  // namespace gpufl
