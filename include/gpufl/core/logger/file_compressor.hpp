#pragma once

#include <string>

namespace gpufl {

class IFileCompressor {
   public:
    virtual ~IFileCompressor() = default;
    /// Compress `path` to `path + ".gz"` and remove (or truncate) the
    /// original. Suited for files we own outright.
    virtual bool compress(const std::string& path) = 0;
    /// Compress `src` into `dst` WITHOUT touching `src` - a pure read of
    /// the source, so it succeeds even while another process holds it.
    virtual bool compressTo(const std::string& src,
                            const std::string& dst) = 0;
};

class GzipFileCompressor final : public IFileCompressor {
   public:
    bool compress(const std::string& path) override;
    bool compressTo(const std::string& src, const std::string& dst) override;
};

}  // namespace gpufl
