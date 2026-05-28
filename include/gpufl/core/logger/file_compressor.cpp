#include "gpufl/core/logger/file_compressor.hpp"

#include <filesystem>
#include <fstream>

#include <zlib.h>

namespace gpufl {
namespace fs = std::filesystem;

bool GzipFileCompressor::compress(const std::string& path) {
    if (path.empty()) return false;

    const std::string outPath = path + ".gz";

    // Scope the input stream tightly so it's closed BEFORE we try to
    // remove the source. On Windows, fs::remove fails with
    // ERROR_SHARING_VIOLATION while any handle to the file is still
    // open — which would leave the uncompressed source next to the
    // new .gz, and the uploader would then read both and upload the
    // same events twice. (Linux's unlink-while-open is forgiving so
    // this only manifested on Windows, but the explicit scope is
    // correct on every platform.)
    bool ok = true;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        gzFile gz = gzopen(outPath.c_str(), "wb");
        if (!gz) return false;

        char buf[65536];
        while (in) {
            in.read(buf, sizeof(buf));
            const auto n = static_cast<unsigned>(in.gcount());
            if (n > 0 && gzwrite(gz, buf, n) != static_cast<int>(n)) {
                ok = false;
                break;
            }
        }
        gzclose(gz);
        // in's dtor runs here, closing the source handle on every
        // platform — safe to fs::remove below.
    }

    std::error_code ec;
    if (ok) {
        fs::remove(path, ec);
    } else {
        fs::remove(outPath, ec);
    }
    return ok;
}

}  // namespace gpufl
