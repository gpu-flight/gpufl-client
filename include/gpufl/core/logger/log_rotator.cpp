#include "gpufl/core/logger/log_rotator.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>

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

std::string LogFileRotator::activePath() const {
    std::ostringstream oss;
    oss << sessionDir() << "/" << opt_.channel_name << ".log";
    return oss.str();
}

std::string LogFileRotator::rotatedPath(std::size_t index) const {
    std::ostringstream oss;
    oss << sessionDir() << "/" << opt_.channel_name << "." << index << ".log";
    return oss.str();
}

void LogFileRotator::rotate() const {
    const std::size_t maxFiles = std::max<std::size_t>(opt_.max_files, 1);
    const std::string active = activePath();

    {
        const std::string oldest = rotatedPath(maxFiles);
        std::error_code ec;
        fs::remove(oldest, ec);
        fs::remove(oldest + ".gz", ec);
    }

    for (std::size_t i = maxFiles; i > 1; --i) {
        const std::string fromBase = rotatedPath(i - 1);
        const std::string toBase = rotatedPath(i);
        if (std::error_code ec; fs::exists(fromBase + ".gz", ec)) {
            fs::rename(fromBase + ".gz", toBase + ".gz", ec);
        } else if (fs::exists(fromBase, ec)) {
            fs::rename(fromBase, toBase, ec);
        }
    }

    {
        const std::string rotated = rotatedPath(1);
        if (std::error_code ec; fs::exists(active, ec)) {
            fs::rename(active, rotated, ec);
            if (opt_.compress_rotated && compressor_ && fs::exists(rotated, ec)) {
                compressor_->compress(rotated);
            }
        }
    }
}

void LogFileRotator::compressActive() const {
    // Defensive guards. Each branch is a no-op rather than an error —
    // compress-on-shutdown is best-effort: a session that crashed
    // before opening any channel, or that opted out of compression,
    // is still a valid session and shouldn't surface failure.
    if (!opt_.compress_rotated) return;
    if (!compressor_) return;

    const std::string active = activePath();
    std::error_code ec;
    if (!fs::exists(active, ec)) return;

    // Empty file → nothing useful to compress. Remove so the session
    // dir doesn't leak a zero-byte .log alongside potentially-real
    // .log.gz files from earlier rotations within the same session.
    if (fs::file_size(active, ec) == 0) {
        fs::remove(active, ec);
        return;
    }

    compressor_->compress(active);

    // Belt-and-suspenders: if compress() reports success but the
    // source file is somehow still there (Windows file-locking edge
    // case, async filesystem, etc.), force-remove it. Leaving both
    // <channel>.log and <channel>.log.gz behind would cause the
    // uploader to read both and upload duplicates. Better to lose a
    // few bytes than emit two copies of every event.
    std::error_code rm_ec;
    if (fs::exists(active, rm_ec)) {
        fs::remove(active, rm_ec);
    }
}

}  // namespace gpufl
