#include "gpufl/core/logger/log_rotator.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>

namespace gpufl {
namespace fs = std::filesystem;

LogFileRotator::LogFileRotator(LogRotationOptions opt, IFileCompressor* compressor)
    : opt_(std::move(opt)), compressor_(compressor) {}

std::string LogFileRotator::basePrefix() const {
    fs::path p(opt_.base_path);
    if (p.extension() == ".log") {
        p.replace_extension();
    }
    return p.string();
}

std::string LogFileRotator::activePath() const {
    std::ostringstream oss;
    oss << basePrefix() << "." << opt_.channel_name << ".log";
    return oss.str();
}

std::string LogFileRotator::rotatedPath(std::size_t index) const {
    std::ostringstream oss;
    oss << basePrefix() << "." << opt_.channel_name << "." << index << ".log";
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

}  // namespace gpufl
