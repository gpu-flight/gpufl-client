#include "gpufl/core/logger/file_log_sink.hpp"

#include <filesystem>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/log_rotator.hpp"
#include "gpufl/core/logger/log_salvage.hpp"

namespace gpufl {
namespace fs = std::filesystem;

// --- FileChannel --- (behavior preserved from pre-refactor Logger::LogChannel)

FileLogSink::FileChannel::FileChannel(std::string name, Logger::Options opt)
    : name_(std::move(name)), opt_(std::move(opt)) {
    if (opt_.compress_rotated) {
        compressor_ = std::make_unique<GzipFileCompressor>();
    }
    LogRotationOptions r{};
    r.base_path = opt_.base_path;
    r.session_id = opt_.session_id;
    r.channel_name = name_;
    r.max_files = opt_.max_files;
    r.compress_rotated = opt_.compress_rotated;
    rotator_ = std::make_unique<LogFileRotator>(r, compressor_.get());

    // v1.2+ requires a session_id. Without it the rotator would write
    // to "<base>/<empty>/channel.log" - a path with an empty path
    // component, which `fs::create_directories` would happily make as
    // just "<base>/", landing files at parent level. The uploader
    // would then flag the parent-level files as the legacy flat
    // layout and refuse to upload. Fail loudly here instead.
    if (!opt_.base_path.empty() && !opt_.session_id.empty()) {
        opened_ = true;
        ensureOpenLocked();
        // ensureOpenLocked logs the real fs error and bails on failure
        // (e.g. EACCES from a Docker volume mount whose existing
        // contents belong to a different UID). Reflect that failure in
        // opened_ so subsequent write() calls short-circuit and so
        // FileLogSink::anyChannelOpen() can report the truth up to
        // Logger::open() → gpufl::init(). Without this flip, init()
        // returned true with a sink that silently dropped every event.
        if (!stream_.is_open()) {
            opened_ = false;
        }
    } else if (!opt_.base_path.empty()) {
        GFL_LOG_ERROR("FileLogSink: session_id is required (got empty). "
                      "Channel '", name_, "' will not be opened.");
    }
}

FileLogSink::FileChannel::~FileChannel() { close(); }

void FileLogSink::FileChannel::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closeLocked();
}

void FileLogSink::FileChannel::closeLocked() {
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    // v1.2+: compress the active .log → .log.gz on clean shutdown so
    // the finished session directory contains only compressed files.
    // The rotator's compressActive() is a no-op when the active file
    // doesn't exist (channel never wrote a byte) or compression is
    // disabled, so it's safe to call unconditionally here.
    //
    // On crash (process killed without a clean Logger::close()), this
    // path doesn't run - the uploader's lazy crash-repair branch
    // gzips the orphan .log on first read instead. That keeps the
    // on-wire format uniform regardless of whether shutdown ran.
    if (rotator_) {
        rotator_->compressActive();
    }
    opened_ = false;
}

bool FileLogSink::FileChannel::isOpen() const { return opened_; }

void FileLogSink::FileChannel::ensureOpenLocked() {
    if (!opened_) return;
    if (stream_.is_open()) return;

    const std::string path = rotator_->activePath();
    const fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        // Surface the actual fs error. The most common failure here
        // is EACCES on a pre-existing log_path mounted from the host
        // (or carried over from a prior container image build) whose
        // contents are owned by a UID the current container can't
        // write to. Without this message, init() either crashed
        // downstream or silently dropped every event with no clue
        // why. Return early so we don't immediately try to open() a
        // file in a directory we know doesn't exist.
        if (ec) {
            GFL_LOG_ERROR(
                "Failed to create log directory '",
                p.parent_path().string(), "' for channel '", name_,
                "': ", ec.message(), " (error code ", ec.value(), "). ",
                "Common causes: directory exists with restrictive "
                "permissions (e.g. a Docker volume mount from a "
                "previous container image - chown the path to the "
                "current container's UID, or delete the directory "
                "and let gpufl recreate it), read-only filesystem, "
                "out of inodes.");
            return;
        }
    }
    stream_.open(p, std::ios::out | std::ios::app);
    if (!stream_.good()) {
        GFL_LOG_ERROR(
            "Failed to open log file '", p.string(), "' for channel '",
            name_, "': ofstream not in a good state after open(). ",
            "Check that the parent directory exists and is writable "
            "by the current process UID.");
    } else {
        std::error_code ec;
        current_bytes_ = fs::exists(p, ec) ? fs::file_size(p, ec) : 0;
    }
}

void FileLogSink::FileChannel::rotateLocked() {
    if (!opened_) return;
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    rotator_->rotate();
    current_bytes_ = 0;
    ensureOpenLocked();
}

void FileLogSink::FileChannel::write(std::string_view line) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!opened_) {
        GFL_LOG_ERROR("Write failed: Channel '", name_, "' is not opened");
        return;
    }
    ensureOpenLocked();
    if (!stream_.good()) {
        GFL_LOG_ERROR("Write failed: Stream bad for '", name_, "'");
        return;
    }
    const size_t bytesToWrite = line.size() + 1;
    if (opt_.rotate_bytes > 0 &&
        (current_bytes_ + bytesToWrite) > opt_.rotate_bytes) {
        rotateLocked();
        if (!stream_.good()) {
            GFL_LOG_ERROR("Write failed after rotate for '", name_, "'");
            return;
        }
    }
    // Write line + newline. Per-write flush is gated behind
    // Options::flush_always (wired from InitOptions::flush_logs_always)
    // because it's a real syscall in the hot path and matters under
    // high event volume - SASS-heavy PyTorch sessions can push tens of
    // thousands of batched lines per second across channels, at which
    // point the fsync cost dominates.
    //
    // When OFF (default): rely on libc's ofstream buffer (~4 KiB) +
    // OS page cache. Rotation (rotateLocked, above) and shutdown
    // (closeLocked) explicitly flush, so the .log → .log.gz pipeline
    // and the uploader's lazy crash-repair path both still see complete
    // data. Worst case on SIGKILL: a few buffered NDJSON lines lost -
    // acceptable for profiling data.
    //
    // When ON: flush at every line boundary so a live tailer (the agent's
    // LogTailer in the Phase 4 live-push design) never observes a
    // partial mid-record line. Without this, the tailer's readLine() can
    // hit a buffered-but-unflushed write, get truncated JSON, and Jackson
    // throws FAIL_ON_TRAILING_TOKENS errors.
    stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream_.put('\n');
    if (opt_.flush_always) {
        stream_.flush();
    }
    current_bytes_ += bytesToWrite;
}

// --- FileLogSink ---

FileLogSink::FileLogSink(const Logger::Options& opt) {
    if (opt.base_path.empty()) return;
    chanDevice_ = std::make_unique<FileChannel>("device", opt);
    chanScope_  = std::make_unique<FileChannel>("scope",  opt);
    chanSystem_ = std::make_unique<FileChannel>("system", opt);
    LogRotationOptions r{};
    r.base_path = opt.base_path;
    r.session_id = opt.session_id;
    temp_dir_ = LogFileRotator(r, nullptr).tempDir();
}

FileLogSink::~FileLogSink() { close(); }

bool FileLogSink::anyChannelOpen() const {
    return (chanDevice_ && chanDevice_->isOpen()) ||
           (chanScope_  && chanScope_->isOpen())  ||
           (chanSystem_ && chanSystem_->isOpen());
}

void FileLogSink::close() {
    if (chanDevice_) chanDevice_->close();
    if (chanScope_)  chanScope_->close();
    if (chanSystem_) chanSystem_->close();
    chanDevice_.reset();
    chanScope_.reset();
    chanSystem_.reset();
    // Every channel has exported and dropped its active file - the
    // shared session temp dir can go now (loud on failure, swept by the
    // launcher's salvage pass otherwise).
    if (!temp_dir_.empty()) {
        const fs::path session_dir = fs::path(temp_dir_).parent_path();
        const auto salvage = salvageSessionTempDir(session_dir);
        if (salvage.deferred > 0 || sessionTempDirHasDeferredData(session_dir)) {
            GFL_LOG_ERROR("[Logger] session temp dir '", temp_dir_,
                          "' still contains deferred log data (salvaged=",
                          salvage.salvaged, ", deferred=", salvage.deferred,
                          ") - leaving it for the next salvage pass.");
            temp_dir_.clear();
            return;
        }
        std::error_code ec;
        fs::remove_all(temp_dir_, ec);
        if (ec) {
            GFL_LOG_ERROR("[Logger] could not remove session temp dir '",
                          temp_dir_, "' (", ec.message(),
                          ") - safe to delete once no process holds it.");
        }
        temp_dir_.clear();
    }
}

FileLogSink::FileChannel* FileLogSink::resolveChannel(Channel ch) const {
    switch (ch) {
        case Channel::Device: return chanDevice_.get();
        case Channel::Scope:  return chanScope_.get();
        case Channel::System: return chanSystem_.get();
        default:              return nullptr;
    }
}

void FileLogSink::write(Channel ch, std::string_view json) {
    if (ch == Channel::All) {
        if (chanDevice_) chanDevice_->write(json);
        if (chanScope_)  chanScope_->write(json);
        if (chanSystem_) chanSystem_->write(json);
    } else {
        if (FileChannel* channel = resolveChannel(ch)) {
            channel->write(json);
        }
    }
}

}  // namespace gpufl
