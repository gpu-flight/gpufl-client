#include "gpufl/core/logger/file_log_sink.hpp"

#include <filesystem>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/log_rotator.hpp"

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
    // to "<base>/<empty>/channel.log" — a path with an empty path
    // component, which `fs::create_directories` would happily make as
    // just "<base>/", landing files at parent level. The uploader
    // would then flag the parent-level files as the legacy flat
    // layout and refuse to upload. Fail loudly here instead.
    if (!opt_.base_path.empty() && !opt_.session_id.empty()) {
        opened_ = true;
        ensureOpenLocked();
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
    // path doesn't run — the uploader's lazy crash-repair branch
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
    }
    stream_.open(p, std::ios::out | std::ios::app);
    if (!stream_.good()) {
        GFL_LOG_ERROR("Failed to open log file: ", p.string());
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
    // Write line + newline, then flush so the agent's LogTailer never
    // observes a partial (mid-record) line. Without this, the tailer can
    // call readLine() on a buffered-but-unflushed write, get a truncated
    // JSON, and Jackson throws FAIL_ON_TRAILING_TOKENS errors.
    stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream_.put('\n');
    stream_.flush();  // Always flush at line boundary — writes are already
                      // rare (one per batched event), cost is negligible.
    current_bytes_ += bytesToWrite;
}

// --- FileLogSink ---

FileLogSink::FileLogSink(const Logger::Options& opt) {
    if (opt.base_path.empty()) return;
    chanDevice_ = std::make_unique<FileChannel>("device", opt);
    chanScope_  = std::make_unique<FileChannel>("scope",  opt);
    chanSystem_ = std::make_unique<FileChannel>("system", opt);
}

FileLogSink::~FileLogSink() { close(); }

void FileLogSink::close() {
    if (chanDevice_) chanDevice_->close();
    if (chanScope_)  chanScope_->close();
    if (chanSystem_) chanSystem_->close();
    chanDevice_.reset();
    chanScope_.reset();
    chanSystem_.reset();
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
