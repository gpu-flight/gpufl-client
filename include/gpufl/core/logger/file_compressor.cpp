#include "gpufl/core/logger/file_compressor.hpp"

#include <filesystem>
#include <fstream>

#include <zlib.h>

namespace gpufl {
namespace fs = std::filesystem;

bool GzipFileCompressor::compress(const std::string& path) {
    if (path.empty()) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    const std::string outPath = path + ".gz";
    gzFile gz = gzopen(outPath.c_str(), "wb");
    if (!gz) return false;

    char buf[65536];
    bool ok = true;
    while (in) {
        in.read(buf, sizeof(buf));
        const auto n = static_cast<unsigned>(in.gcount());
        if (n > 0 && gzwrite(gz, buf, n) != static_cast<int>(n)) {
            ok = false;
            break;
        }
    }

    gzclose(gz);

    if (ok) {
        std::error_code ec;
        fs::remove(path, ec);
    } else {
        std::error_code ec;
        fs::remove(outPath, ec);
    }
    return ok;
}

}  // namespace gpufl
