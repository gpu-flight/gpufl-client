#include "common/log_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include <zlib.h>

#include "gpufl/core/json/json.hpp"

namespace fs = std::filesystem;

namespace gpufl::test {

fs::path MakeTempLogDir() {
    // Counter ensures uniqueness even if the wall clock has low resolution.
    static std::atomic<int> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const int n = counter.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream name;
    name << "gpufl_test_" << now << "_" << n;

    const fs::path dir = fs::temp_directory_path() / name.str();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("MakeTempLogDir: create_directories failed: " +
                                 ec.message());
    }
    return dir;
}

namespace {

// v1.2 layout — files live at `<dir>/<prefix>/<session_id>/<channel>.log`
// or `<dir>/<prefix>/<session_id>/<channel>.<N>.log[.gz]`. The test
// helpers were originally written for the v1.1 flat layout
// (`<dir>/<prefix>.<channel>.log`); this walks the per-session
// subdirectories instead, aggregating across every session inside the
// log_path. Tests that ran one session see one session's worth of
// data; tests that ran multiple see them all merged.
std::vector<fs::path> FindChannelFiles(const fs::path& dir,
                                       const std::string& prefix,
                                       const std::string& channel) {
    std::vector<fs::path> out;
    // <dir>/<prefix> is the log_path the rotator was given. Each
    // subdirectory inside it is a session.
    const fs::path log_path = dir / prefix;
    if (!fs::exists(log_path)) return out;

    const std::regex channel_pattern(
        "^" + channel + R"((?:\.\d+)?\.log(?:\.gz)?$)");

    for (const auto& session_entry : fs::directory_iterator(log_path)) {
        if (!session_entry.is_directory()) continue;
        const std::string sid = session_entry.path().filename().string();
        if (sid.empty() || sid.front() == '.') continue;

        for (const auto& entry : fs::directory_iterator(session_entry.path())) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, channel_pattern)) {
                out.push_back(entry.path());
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// Read a NDJSON file (plain or gzipped) into JsonValue records.
/// Gzipped files are transparently decompressed — needed because
/// v1.2's compress-on-shutdown turns the active .log into .log.gz on
/// clean shutdown, so test code that runs gpufl::shutdown() and then
/// reads back the logs encounters .log.gz files.
std::vector<JsonValue> readNdjsonAny(const fs::path& path) {
    std::vector<JsonValue> out;
    const bool gz = path.extension() == ".gz";
    if (!gz) {
        return ::gpufl::json::loadJsonLines(path.string());
    }
    // Streaming gunzip via zlib's gzgets.
    gzFile f = gzopen(path.string().c_str(), "rb");
    if (!f) return out;
    std::string line;
    char buf[8192];
    while (gzgets(f, buf, sizeof(buf)) != nullptr) {
        line.append(buf);
        if (line.empty()) continue;
        if (line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) {
                try { out.push_back(::gpufl::json::parseJson(line)); }
                catch (...) { /* skip malformed */ }
            }
            line.clear();
        }
    }
    if (!line.empty()) {
        try { out.push_back(::gpufl::json::parseJson(line)); }
        catch (...) {}
    }
    gzclose(f);
    return out;
}

std::vector<JsonValue> ReadChannel(const fs::path& dir,
                                   const std::string& prefix,
                                   const std::string& channel) {
    std::vector<JsonValue> merged;
    for (const auto& file : FindChannelFiles(dir, prefix, channel)) {
        auto records = readNdjsonAny(file);
        merged.insert(merged.end(),
                      std::make_move_iterator(records.begin()),
                      std::make_move_iterator(records.end()));
    }
    return merged;
}

}  // namespace

LogEvents ReadAllLogs(const fs::path& dir, const std::string& prefix) {
    return LogEvents{
        ReadChannel(dir, prefix, "device"),
        ReadChannel(dir, prefix, "scope"),
        ReadChannel(dir, prefix, "system"),
    };
}

std::vector<JsonValue> FilterByType(const std::vector<JsonValue>& events,
                                    const std::string& type) {
    std::vector<JsonValue> out;
    for (const auto& ev : events) {
        if (!ev.is_object()) continue;
        if (ev.value<std::string>("type", "") == type) {
            out.push_back(ev);
        }
    }
    return out;
}

int CountProfileSamplesOfKind(const std::vector<JsonValue>& device_events,
                              const std::string& kind) {
    int total = 0;
    for (const auto& batch : FilterByType(device_events, "profile_sample_batch")) {
        if (!batch.contains("columns") || !batch.contains("rows")) continue;
        const auto& cols = batch["columns"];
        const auto& rows = batch["rows"];
        if (!cols.is_array() || !rows.is_array()) continue;

        // Find the index of the sample_kind column.
        int kindIdx = -1;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i].is_string() &&
                cols[i].get_string() == "sample_kind") {
                kindIdx = static_cast<int>(i);
                break;
            }
        }
        if (kindIdx < 0) continue;

        for (size_t r = 0; r < rows.size(); ++r) {
            const auto& row = rows[r];
            if (!row.is_array() || row.size() <= static_cast<size_t>(kindIdx)) {
                continue;
            }
            const auto& cell = row[static_cast<size_t>(kindIdx)];
            // sample_kind may appear as a string ("pc_sampling") or as
            // the numeric discriminator (0 = pc_sampling, 1 = sass_metric)
            // depending on the emitter.
            if (cell.is_string()) {
                if (cell.get_string() == kind) ++total;
            } else if (cell.is_number()) {
                const int64_t v = cell.as_int(-1);
                if ((kind == "pc_sampling" && v == 0) ||
                    (kind == "sass_metric" && v == 1)) {
                    ++total;
                }
            }
        }
    }
    return total;
}

std::vector<std::string> GetStringArrayField(
    const std::vector<JsonValue>& events, const std::string& type,
    const std::string& field) {
    std::vector<std::string> out;
    auto matched = FilterByType(events, type);
    if (matched.empty()) return out;
    const auto& ev = matched.front();
    if (!ev.contains(field) || !ev[field].is_array()) return out;
    const auto& arr = ev[field].get_array();
    for (const auto& v : arr) {
        if (v.is_string()) out.push_back(v.get_string());
    }
    return out;
}

const char* EngineName(gpufl::ProfilingEngine e) {
    switch (e) {
        case gpufl::ProfilingEngine::None:               return "None";
        case gpufl::ProfilingEngine::PcSampling:         return "PcSampling";
        case gpufl::ProfilingEngine::SassMetrics:        return "SassMetrics";
        case gpufl::ProfilingEngine::RangeProfiler:      return "RangeProfiler";
        case gpufl::ProfilingEngine::PcSamplingWithSass: return "PcSamplingWithSass";
    }
    return "Unknown";
}

}  // namespace gpufl::test
