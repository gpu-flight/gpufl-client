#pragma once

#include <string>

namespace gpufl {

class IFileCompressor {
   public:
    virtual ~IFileCompressor() = default;
    virtual bool compress(const std::string& path) = 0;
};

class GzipFileCompressor final : public IFileCompressor {
   public:
    bool compress(const std::string& path) override;
};

}  // namespace gpufl
