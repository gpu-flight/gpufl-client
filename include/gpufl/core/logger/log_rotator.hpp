#pragma once

#include <cstddef>
#include <string>

#include "gpufl/core/logger/file_compressor.hpp"

namespace gpufl {

struct LogRotationOptions {
    std::string base_path;
    std::string channel_name;
    std::size_t max_files = 100;
    bool compress_rotated = true;
};

class LogFileRotator {
   public:
    LogFileRotator(LogRotationOptions opt, IFileCompressor* compressor);

    [[nodiscard]] std::string activePath() const;
    void rotate()const;

   private:
    [[nodiscard]] std::string basePrefix() const;
    [[nodiscard]] std::string rotatedPath(std::size_t index) const;

    LogRotationOptions opt_;
    IFileCompressor* compressor_ = nullptr;  // non-owning
};

}  // namespace gpufl
