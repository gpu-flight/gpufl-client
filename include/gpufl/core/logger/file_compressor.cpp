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

bool removeOrTruncateFile(const std::string& path) {
    if (path.empty()) return true;

    const fs::path p(path);
    std::error_code ec;
    if (removeWithRetry(p, ec)) return true;

    // Windows can deny delete sharing while still allowing the owner to
    // truncate. An empty husk is harmless and the temp-dir cleanup removes it
    // once the holder lets go.
    std::ofstream trunc(p, std::ios::out | std::ios::trunc);
    if (!trunc) return false;
    trunc.close();

    std::error_code size_ec;
    return fs::file_size(p, size_ec) == 0 && !size_ec;
}

bool GzipFileCompressor::compressTo(const std::string& src,
                                    const std::string& dst) {
    if (src.empty() || dst.empty()) return false;

    std::ifstream in(src, std::ios::binary);
    if (!in) return false;

    gzFile gz = gzopen(dst.c_str(), "wb");
    if (!gz) return false;

    bool ok = true;
    char buf[65536];
    while (in) {
        in.read(buf, sizeof(buf));
        const auto n = static_cast<unsigned>(in.gcount());
        if (n > 0 && gzwrite(gz, buf, n) != static_cast<int>(n)) {
            ok = false;
            break;
        }
    }
    if (in.bad()) ok = false;
    if (gzclose(gz) != Z_OK) ok = false;
    if (!ok) {
        std::error_code ec;
        fs::remove(dst, ec);
    }
    return ok;
}

bool GzipFileCompressor::compress(const std::string& path) {
    if (path.empty()) return false;

    const std::string outPath = path + ".gz";

    // compressTo only READS the source; its stream is closed before the
    // remove below. On Windows, fs::remove fails with
    // ERROR_SHARING_VIOLATION while any other handle to the file is
    // open - which would leave the uncompressed source next to the new
    // .gz, and the uploader would then read both and upload the same
    // events twice.
    const bool ok = compressTo(path, outPath);
    if (!ok) return false;

    if (!removeOrTruncateFile(path)) {
        // Keep the source authoritative when the exact-once transition could
        // not complete. A caller seeing false may retry safely; leaving both
        // a non-empty raw file and a successful gzip would invite duplicate
        // salvage/upload.
        std::error_code rm_ec;
        fs::remove(outPath, rm_ec);
        GFL_LOG_ERROR("[Logger] compressed '", path,
                      "' but could not remove or truncate the original; "
                      "discarded the gzip and kept the raw source.");
        return false;
    }
    return true;
}

}  // namespace gpufl
