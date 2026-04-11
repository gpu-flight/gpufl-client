#include "gpufl/core/logger/logger.hpp"

#include <filesystem>
#include <memory>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/log_rotator.hpp"

namespace gpufl {
namespace fs = std::filesystem;

// --- LogChannel ---

Logger::LogChannel::LogChannel(std::string name, Options opt)
    : name_(std::move(name)), opt_(std::move(opt)) {
    if (opt_.compress_rotated) {
        compressor_ = std::make_unique<GzipFileCompressor>();
    }
    LogRotationOptions r{};
    r.base_path = opt_.base_path;
    r.channel_name = name_;
    r.max_files = opt_.max_files;
    r.compress_rotated = opt_.compress_rotated;
    rotator_ = std::make_unique<LogFileRotator>(r, compressor_.get());

    if (!opt_.base_path.empty()) {
        opened_ = true;
        ensureOpenLocked();
    }
}

Logger::LogChannel::~LogChannel() { close(); }

void Logger::LogChannel::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closeLocked();
}

void Logger::LogChannel::closeLocked() {
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    opened_ = false;
}

bool Logger::LogChannel::isOpen() const { return opened_; }

void Logger::LogChannel::ensureOpenLocked() {
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

void Logger::LogChannel::rotateLocked() {
    if (!opened_) return;
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    rotator_->rotate();
    current_bytes_ = 0;
    ensureOpenLocked();
}

void Logger::LogChannel::write(const std::string& line) {
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
    // Write line + newline in a single buffer so the agent never sees a
    // partial line on disk.  Always flush to ensure the complete line is
    // visible to the tailing agent before the next write begins.
    stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream_.put('\n');
    stream_.flush();
    current_bytes_ += bytesToWrite;
}

// --- Logger ---

Logger::Logger() = default;
Logger::~Logger() { close(); }

bool Logger::open(const Options& opt) {
    close();
    opt_ = opt;
    if (opt_.base_path.empty()) return false;
    chanDevice_ = std::make_unique<LogChannel>("device", opt_);
    chanScope_  = std::make_unique<LogChannel>("scope",  opt_);
    chanSystem_ = std::make_unique<LogChannel>("system", opt_);
    return true;
}

void Logger::close() {
    if (chanDevice_) chanDevice_->close();
    if (chanScope_)  chanScope_->close();
    if (chanSystem_) chanSystem_->close();
    chanDevice_.reset();
    chanScope_.reset();
    chanSystem_.reset();
}

Logger::LogChannel* Logger::resolveChannel(Channel ch)const {
    switch (ch) {
        case Channel::Device: return chanDevice_.get();
        case Channel::Scope:  return chanScope_.get();
        case Channel::System: return chanSystem_.get();
        default:              return nullptr;
    }
}

void Logger::write(const IJsonSerializable& model) {
    const std::string json = model.buildJson();
    if (model.channel() == Channel::All) {
        if (chanDevice_) chanDevice_->write(json);
        if (chanScope_)  chanScope_->write(json);
        if (chanSystem_) chanSystem_->write(json);
    } else {
        if (LogChannel* ch = resolveChannel(model.channel())) {
            ch->write(json);
        }
    }
}

}  // namespace gpufl
