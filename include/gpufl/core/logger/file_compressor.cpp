#include "gpufl/core/logger/file_compressor.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <zlib.h>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {
namespace fs = std::filesystem;

namespace {
/// Remove with a short backoff (100/200/400 ms). On Windows a freshly
/// written file is frequently held for a moment by an AV scan, the
/// search indexer, or a tailing reader - a single immediate remove
/// loses that race and leaves a stale .log next to its .gz. True on
/// success OR when the file is already gone.
bool removeWithRetry(const fs::path& p, std::error_code& ec) {
    for (int attempt = 0;; ++attempt) {
        if (fs::remove(p, ec) || !ec) return true;
        if (attempt >= 2) return false;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100 << attempt));
    }
}
}  // namespace

bool GzipFileCompressor::compress(const std::string& path) {
    if (path.empty()) return false;

    const std::string outPath = path + ".gz";

    // Scope the input stream tightly so it's closed BEFORE we try to
    // remove the source. On Windows, fs::remove fails with
    // ERROR_SHARING_VIOLATION while any handle to the file is still
    // open - which would leave the uncompressed source next to the
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
        // platform - safe to fs::remove below.
    }

    std::error_code ec;
    if (ok) {
        if (!removeWithRetry(path, ec)) {
            // A holder outlived the retries (a tail, an editor, an AV
            // scan, an uploader). The data is safe - the .gz is
            // complete - but a stale .log now sits next to it; the
            // launcher repair and the uploader's discovery both remove
            // it later. Log it so the leftover is explainable.
            GFL_LOG_ERROR("[Logger] compressed '", path,
                          "' but could not remove the original (",
                          ec.message(),
                          ") - stale .log left next to the .gz.");
        }
    } else {
        fs::remove(outPath, ec);
    }
    return ok;
}

}  // namespace gpufl
