#include "gpufl/report/text_report.hpp"
#include "gpufl/report/hint_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpufl {
namespace report {

namespace fs = std::filesystem;

// ── Formatting helpers (file-local) ─────────────────────────────────────────

namespace {

const std::string SEP(79, '=');

std::string fmtBytes(uint64_t n) {
    std::ostringstream oss;
    if (n == 0)               return "0 B";
    if (n >= 1024ULL * 1024)  { oss << std::fixed << std::setprecision(1) << (n / 1048576.0) << " MB"; return oss.str(); }
    if (n >= 1024)            { oss << std::fixed << std::setprecision(1) << (n / 1024.0) << " KB"; return oss.str(); }
    return std::to_string(n) + " B";
}

std::string fmtDuration(double ms) {
    std::ostringstream oss;
    if (ms >= 1000.0)      oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
    else if (ms >= 1.0)    oss << std::fixed << std::setprecision(2) << ms << " ms";
    else                   oss << std::fixed << std::setprecision(2) << (ms * 1000.0) << " us";
    return oss.str();
}

std::string fmtPower(double mw) {
    std::ostringstream oss;
    if (mw >= 1000.0) oss << std::fixed << std::setprecision(1) << (mw / 1000.0) << " W";
    else              oss << std::fixed << std::setprecision(0) << mw << " mW";
    return oss.str();
}

std::string shortenKernelName(const std::string& name) {
    std::string s = name;
    auto at = s.find('@');
    if (at != std::string::npos) s = s.substr(0, at);
    for (const auto* prefix : {"void ", "int ", "float ", "double ", "__global__ "}) {
        if (s.rfind(prefix, 0) == 0) { s = s.substr(std::string(prefix).size()); break; }
    }
    auto tpl = s.find('<');
    std::string funcPart = (tpl != std::string::npos) ? s.substr(0, tpl) : s;
    auto paren = funcPart.find('(');
    if (paren != std::string::npos) funcPart = funcPart.substr(0, paren);

    std::vector<std::string> parts;
    std::istringstream iss(funcPart);
    std::string part;
    while (std::getline(iss, part, ':'))
        if (!part.empty()) parts.push_back(part);

    std::string shortFunc = (parts.size() > 2)
        ? parts[parts.size() - 2] + "::" + parts.back()
        : funcPart;
    if (tpl != std::string::npos) shortFunc += "<...>";
    return shortFunc;
}

std::string truncate(const std::string& s, size_t maxLen) {
    return (s.size() > maxLen) ? s.substr(0, maxLen - 3) + "..." : s;
}

const std::map<int, std::string> kCopyKindNames = {
    {1, "HtoD"}, {2, "DtoH"}, {3, "HtoA"}, {4, "AtoH"},
    {5, "AtoA"}, {6, "AtoD"}, {7, "DtoA"}, {8, "DtoD"},
    {9, "HtoH"}, {10, "PtoP"},
};

std::string resolveCopyKind(int kind) {
    auto it = kCopyKindNames.find(kind);
    return (it != kCopyKindNames.end()) ? it->second : "Unknown(" + std::to_string(kind) + ")";
}

// Values match CUpti_ActivityPCSamplingStallReason enum from cupti_activity.h
const std::map<int, std::string> kStallNames = {
    {2,  "Instruction Fetch"},       {3,  "Execution Dependency"},
    {4,  "Memory Dependency"},       {5,  "Texture"},
    {6,  "Sync"},                    {7,  "Constant Memory"},
    {8,  "Pipe Busy"},               {9,  "Memory Throttle"},
    {10, "Not Selected"},            {11, "Other"},
    {12, "Sleeping"},
};

std::string resolveStallReason(int reason) {
    auto it = kStallNames.find(reason);
    return (it != kStallNames.end()) ? it->second : "Stall_" + std::to_string(reason);
}

// Resolve log file path for a given channel
std::string resolveLogPath(const std::string& dir, const std::string& prefix,
                           const std::string& channel) {
    std::string base = prefix;
    if (base.size() > 4 && base.substr(base.size() - 4) == ".log")
        base = base.substr(0, base.size() - 4);

    fs::path active = fs::path(dir) / (base + "." + channel + ".log");
    if (fs::exists(active)) return active.string();

    std::vector<std::pair<int, std::string>> candidates;
    std::error_code ec;
    for (auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        const std::string pat = base + "." + channel + ".";
        if (name.rfind(pat, 0) == 0) {
            std::string rest = name.substr(pat.size());
            auto dotPos = rest.find(".log");
            if (dotPos != std::string::npos) {
                try { candidates.emplace_back(std::stoi(rest.substr(0, dotPos)), entry.path().string()); }
                catch (...) {}
            }
        }
    }
    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end());
        return candidates.front().second;
    }
    return {};
}

// Batch row helpers
using ColIndex = std::unordered_map<std::string, size_t>;

ColIndex buildColumnIndex(const JsonValue& cols) {
    ColIndex ci;
    const auto& arr = cols.get_array();
    for (size_t i = 0; i < arr.size(); ++i)
        ci[arr[i].get_string()] = i;
    return ci;
}

int64_t rowInt(const JsonValue& row, const ColIndex& ci, const std::string& col, int64_t def = 0) {
    auto it = ci.find(col);
    return (it != ci.end()) ? row[it->second].as_int(def) : def;
}

uint64_t rowU64(const JsonValue& row, const ColIndex& ci, const std::string& col, uint64_t def = 0) {
    auto it = ci.find(col);
    return (it != ci.end()) ? row[it->second].as_uint64(def) : def;
}

// Functional: sort a map by value descending, return top-N as vector of pairs
template <typename K, typename V, typename Fn>
std::vector<std::pair<K, V>> sortedTopN(const std::map<K, V>& m, int n,
                                         Fn valueFn) {
    std::vector<std::pair<K, V>> vec(m.begin(), m.end());
    std::sort(vec.begin(), vec.end(),
              [&](const auto& a, const auto& b) { return valueFn(a.second) > valueFn(b.second); });
    if (n > 0 && static_cast<int>(vec.size()) > n) vec.resize(n);
    return vec;
}

}  // namespace

// ── Constructor: parse log files ────────────────────────────────────────────

TextReport::TextReport(const Options& opts) : top_n_(opts.top_n) {
    parseLogFiles(opts);
}

void TextReport::parseLogFiles(const Options& opts) {
    auto deviceRecords = loadJsonLines(resolveLogPath(opts.log_dir, opts.log_prefix, "device"));
    auto scopeRecords  = loadJsonLines(resolveLogPath(opts.log_dir, opts.log_prefix, "scope"));
    auto systemRecords = loadJsonLines(resolveLogPath(opts.log_dir, opts.log_prefix, "system"));

    // Find the latest session_id (last job_start across all channels)
    std::string latestSessionId;
    auto findLatestSession = [&](const std::vector<JsonValue>& records) {
        for (const auto& rec : records) {
            std::string type = rec.value<std::string>("type", "");
            if (type == "job_start" || type == "init") {
                std::string sid = rec.value<std::string>("session_id", "");
                if (!sid.empty()) latestSessionId = sid;
            }
        }
    };
    for (const auto* recs : {&deviceRecords, &scopeRecords, &systemRecords})
        findLatestSession(*recs);

    // Filter records to only the latest session
    auto filterBySession = [&](std::vector<JsonValue>& records) {
        if (latestSessionId.empty()) return;
        records.erase(
            std::remove_if(records.begin(), records.end(), [&](const JsonValue& rec) {
                std::string sid = rec.value<std::string>("session_id", "");
                // Keep records that match the latest session or have no session_id
                return !sid.empty() && sid != latestSessionId;
            }),
            records.end());
    };
    filterBySession(deviceRecords);
    filterBySession(scopeRecords);
    filterBySession(systemRecords);

    // Pass 1: dictionaries (needed before batch expansion)
    std::unordered_map<int, std::string> kernel_dict, scope_name_dict, function_dict, metric_dict;
    for (const auto* recs : {&deviceRecords, &scopeRecords, &systemRecords})
        parseDictionaries(*recs, kernel_dict, scope_name_dict, function_dict, metric_dict);

    // Pass 2: typed records
    parseDeviceLog(deviceRecords, kernel_dict);
    parseScopeLog(scopeRecords, scope_name_dict, function_dict, metric_dict);
    parseSystemLog(systemRecords);
}

void TextReport::parseDictionaries(const std::vector<JsonValue>& records,
                                   std::unordered_map<int, std::string>& kernel_dict,
                                   std::unordered_map<int, std::string>& scope_name_dict,
                                   std::unordered_map<int, std::string>& function_dict,
                                   std::unordered_map<int, std::string>& metric_dict) {
    auto merge = [](const JsonValue& obj, std::unordered_map<int, std::string>& dict) {
        if (!obj.is_object()) return;
        for (auto& [k, v] : obj)
            dict[std::stoi(k)] = v.get_string();
    };

    for (const auto& rec : records) {
        if (rec.value<std::string>("type", "") != "dictionary_update") continue;
        merge(rec["kernel_dict"], kernel_dict);
        merge(rec["scope_name_dict"], scope_name_dict);
        merge(rec["function_dict"], function_dict);
        merge(rec["metric_dict"], metric_dict);
    }
}

void TextReport::parseJobStart(const JsonValue& rec, SessionInfo& info) {
    info.app_name = rec.value<std::string>("app", "");
    info.session_id = rec.value<std::string>("session_id", "");
    info.start_ns = rec.value<int64_t>("ts_ns", 0);

    for (const char* field : {"gpu_static_devices", "cuda_static_devices", "rocm_static_devices"}) {
        if (!rec.contains(field) || !rec[field].is_array() || rec[field].empty()) continue;
        const auto& dev = rec[field][0];
        info.gpu_name = dev.value<std::string>("name", "");
        info.compute_major = dev.contains("compute_capability_major")
            ? dev.value<int>("compute_capability_major", 0) : dev.value<int>("major", 0);
        info.compute_minor = dev.contains("compute_capability_minor")
            ? dev.value<int>("compute_capability_minor", 0) : dev.value<int>("minor", 0);
        info.sm_count = dev.value<int>("multi_processor_count", 0);
        info.shared_mem_per_block = dev.value<int>("shared_mem_per_block", 0);
        info.regs_per_block = dev.value<int>("regs_per_block", 0);
        info.l2_cache_size = dev.value<int>("l2_cache_size", 0);
        break;
    }

    if (info.gpu_name.empty() && rec.contains("devices") &&
        rec["devices"].is_array() && !rec["devices"].empty())
        info.gpu_name = rec["devices"][0].value<std::string>("name", "");
}

void TextReport::parseDeviceLog(const std::vector<JsonValue>& records,
                                const std::unordered_map<int, std::string>& kernel_dict) {
    std::unordered_map<unsigned, JsonValue> details;

    for (const auto& rec : records) {
        const std::string type = rec.value<std::string>("type", "");

        if ((type == "job_start" || type == "init") && info_.app_name.empty()) {
            parseJobStart(rec, info_);
        } else if (type == "kernel_event_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            int64_t base = rec.value<int64_t>("base_time_ns", 0);
            std::string sid = rec.value<std::string>("session_id", "");

            for (const auto& row : rec["rows"].get_array()) {
                int64_t dur_ns = rowInt(row, ci, "duration_ns");
                int kid = static_cast<int>(rowInt(row, ci, "kernel_id"));
                auto it = kernel_dict.find(kid);

                kernels_.push_back({
                    /*name*/       (it != kernel_dict.end()) ? it->second : "kernel_" + std::to_string(kid),
                    /*session_id*/ sid,
                    /*start_ns*/   base + rowInt(row, ci, "dt_ns"),
                    /*end_ns*/     base + rowInt(row, ci, "dt_ns") + dur_ns,
                    /*duration_ms*/dur_ns / 1e6,
                    /*stream_id*/  static_cast<uint32_t>(rowInt(row, ci, "stream_id")),
                    /*corr_id*/    static_cast<unsigned>(rowInt(row, ci, "corr_id")),
                    /*num_regs*/   static_cast<int>(rowInt(row, ci, "num_regs")),
                    /*dyn_shared*/ static_cast<int>(rowInt(row, ci, "dyn_shared")),
                    /*has_details*/rowInt(row, ci, "has_details") != 0,
                });
            }
        } else if (type == "kernel_detail") {
            details[rec.value<unsigned>("corr_id", 0u)] = rec;
        } else if (type == "memcpy_event_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            int64_t base = rec.value<int64_t>("base_time_ns", 0);
            for (const auto& row : rec["rows"].get_array()) {
                int64_t dur_ns = rowInt(row, ci, "duration_ns");
                memcpy_.push_back({
                    base + rowInt(row, ci, "dt_ns"),
                    dur_ns / 1e6,
                    rowU64(row, ci, "bytes"),
                    static_cast<int>(rowInt(row, ci, "copy_kind")),
                });
            }
        } else if (type == "shutdown" && rec.contains("ts_ns")) {
            info_.end_ns = rec["ts_ns"].as_int();
        }
    }

    mergeKernelDetails(details);
}

void TextReport::mergeKernelDetails(std::unordered_map<unsigned, JsonValue>& details) {
    for (auto& k : kernels_) {
        auto it = details.find(k.corr_id);
        if (it == details.end()) continue;
        const auto& d = it->second;
        k.grid              = d.value<std::string>("grid", "?");
        k.block             = d.value<std::string>("block", "?");
        k.occupancy         = d.value<float>("occupancy", -1.0f);
        k.reg_occupancy     = d.value<float>("reg_occupancy", -1.0f);
        k.smem_occupancy    = d.value<float>("smem_occupancy", -1.0f);
        k.warp_occupancy    = d.value<float>("warp_occupancy", -1.0f);
        k.block_occupancy   = d.value<float>("block_occupancy", -1.0f);
        k.limiting_resource = d.value<std::string>("limiting_resource", "");
        k.static_shared     = d.value<int>("static_shared", 0);
        k.user_scope        = d.value<std::string>("user_scope", "");
    }
}

void TextReport::parseScopeLog(const std::vector<JsonValue>& records,
                               const std::unordered_map<int, std::string>& scope_name_dict,
                               const std::unordered_map<int, std::string>& function_dict,
                               const std::unordered_map<int, std::string>& metric_dict) {
    auto lookupOr = [](const std::unordered_map<int, std::string>& dict, int id,
                       const std::string& prefix) -> std::string {
        auto it = dict.find(id);
        return (it != dict.end()) ? it->second : prefix + std::to_string(id);
    };

    for (const auto& rec : records) {
        const std::string type = rec.value<std::string>("type", "");

        if ((type == "job_start" || type == "init") && info_.app_name.empty()) {
            parseJobStart(rec, info_);
        } else if (type == "scope_event_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            int64_t base = rec.value<int64_t>("base_time_ns", 0);
            for (const auto& row : rec["rows"].get_array()) {
                int name_id = static_cast<int>(rowInt(row, ci, "name_id"));
                scope_events_.push_back({
                    base + rowInt(row, ci, "dt_ns"),
                    rowU64(row, ci, "scope_instance_id"),
                    lookupOr(scope_name_dict, name_id, "scope_"),
                    static_cast<int>(rowInt(row, ci, "event_type")),
                });
            }
        } else if (type == "profile_sample_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            for (const auto& row : rec["rows"].get_array()) {
                int fn_id = static_cast<int>(rowInt(row, ci, "function_id"));
                int met_id = static_cast<int>(rowInt(row, ci, "metric_id"));
                auto fn_it = function_dict.find(fn_id);
                auto met_it = metric_dict.find(met_id);

                // Read reason_name if available in the columnar data
                std::string rname;
                auto rn_it = ci.find("reason_name");
                if (rn_it != ci.end()) {
                    size_t idx = rn_it->second;
                    const auto& arr = row.get_array();
                    if (idx < arr.size() && arr[idx].is_string())
                        rname = arr[idx].get_string();
                }

                profile_samples_.push_back({
                    (fn_it != function_dict.end()) ? fn_it->second : "",
                    (met_it != metric_dict.end()) ? met_it->second : "",
                    rowU64(row, ci, "metric_value"),
                    static_cast<int>(rowInt(row, ci, "stall_reason")),
                    static_cast<int>(rowInt(row, ci, "sample_kind")),
                    std::move(rname),
                });
            }
        }
    }
}

void TextReport::parseSystemLog(const std::vector<JsonValue>& records) {
    for (const auto& rec : records) {
        const std::string type = rec.value<std::string>("type", "");

        if ((type == "job_start" || type == "init") && info_.app_name.empty()) {
            parseJobStart(rec, info_);
        } else if (type == "device_metric_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            int64_t base = rec.value<int64_t>("base_time_ns", 0);
            for (const auto& row : rec["rows"].get_array()) {
                device_metrics_.push_back({
                    base + rowInt(row, ci, "dt_ns"),
                    static_cast<int>(rowInt(row, ci, "gpu_util")),
                    static_cast<int>(rowInt(row, ci, "mem_util")),
                    static_cast<int>(rowInt(row, ci, "temp_c")),
                    static_cast<int>(rowInt(row, ci, "power_mw")),
                    rowU64(row, ci, "used_mib"),
                    static_cast<int>(rowInt(row, ci, "clock_sm")),
                    static_cast<int>(rowInt(row, ci, "fan_speed_pct")),
                    static_cast<int>(rowInt(row, ci, "temp_mem_c")),
                    static_cast<int>(rowInt(row, ci, "temp_junction_c")),
                    static_cast<int>(rowInt(row, ci, "voltage_mv")),
                    rowU64(row, ci, "energy_uj"),
                    static_cast<int>(rowInt(row, ci, "clock_mem")),
                    rowU64(row, ci, "pcie_bw_bps"),
                    rowU64(row, ci, "ecc_corrected"),
                    rowU64(row, ci, "ecc_uncorrected"),
                });
            }
        } else if (type == "host_metric_batch") {
            auto ci = buildColumnIndex(rec["columns"]);
            int64_t base = rec.value<int64_t>("base_time_ns", 0);
            for (const auto& row : rec["rows"].get_array()) {
                host_metrics_.push_back({
                    base + rowInt(row, ci, "dt_ns"),
                    rowInt(row, ci, "cpu_pct_x100") / 100.0,
                    rowU64(row, ci, "ram_used_mib"),
                    rowU64(row, ci, "ram_total_mib"),
                });
            }
        } else if (type == "system_stop" && info_.end_ns == 0 && rec.contains("ts_ns")) {
            info_.end_ns = rec["ts_ns"].as_int();
        }
    }
}

// ── Report generation ───────────────────────────────────────────────────────

std::string TextReport::generate() const {
    std::ostringstream out;
    writeHeader(out);
    writeSessionSummary(out);
    writeKernelSummary(out);
    writeTopKernels(out);
    writeKernelDetails(out);
    writeMemcpySummary(out);
    writeSystemMetrics(out);
    writeScopeSummary(out);
    writeProfileAnalysis(out);
    return out.str();
}

// ── Section writers ─────────────────────────────────────────────────────────

void TextReport::writeHeader(std::ostringstream& out) const {
    out << SEP << "\n"
        << std::setw(52) << "GPU Flight Session Report" << "\n"
        << SEP << "\n";
}

void TextReport::writeSessionSummary(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Session Summary\n" << SEP << "\n";
    out << "  Application:          " << (info_.app_name.empty() ? "unknown" : info_.app_name) << "\n";

    if (!info_.session_id.empty())
        out << "  Session ID:           " << info_.session_id << "\n";
    if (info_.start_ns > 0 && info_.end_ns > 0)
        out << "  Duration:             " << fmtDuration((info_.end_ns - info_.start_ns) / 1e6) << "\n";

    if (!info_.gpu_name.empty()) {
        out << "  GPU Device:           " << info_.gpu_name << "\n";
        if (info_.compute_major > 0)
            out << "    Compute:            " << info_.compute_major << "." << info_.compute_minor << "\n";
        if (info_.sm_count > 0)
            out << "    SMs:                " << info_.sm_count << "\n";
        if (info_.shared_mem_per_block > 0)
            out << "    Shared Mem/Block:   " << fmtBytes(info_.shared_mem_per_block) << "\n";
        if (info_.regs_per_block > 0)
            out << "    Registers/Block:    " << info_.regs_per_block << "\n";
        if (info_.l2_cache_size > 0)
            out << "    L2 Cache:           " << fmtBytes(info_.l2_cache_size) << "\n";
    }
}

void TextReport::writeKernelSummary(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Kernel Execution Summary\n" << SEP << "\n";
    if (kernels_.empty()) { out << "  (No kernel data)\n"; return; }

    // Functional: extract durations, compute stats via algorithms
    std::vector<double> durations(kernels_.size());
    std::transform(kernels_.begin(), kernels_.end(), durations.begin(),
                   [](const KernelRecord& k) { return k.duration_ms; });

    double totalMs = std::accumulate(durations.begin(), durations.end(), 0.0);
    auto [minIt, maxIt] = std::minmax_element(durations.begin(), durations.end());
    std::sort(durations.begin(), durations.end());

    std::map<std::string, int> uniqueNames;
    for (const auto& k : kernels_) uniqueNames[k.name]++;

    out << "  Total Kernels:        " << kernels_.size() << "\n";
    out << "  Unique Kernels:       " << uniqueNames.size() << "\n";
    out << "  Total GPU Time:       " << fmtDuration(totalMs) << "\n";

    if (info_.start_ns > 0 && info_.end_ns > 0) {
        double sessionMs = (info_.end_ns - info_.start_ns) / 1e6;
        if (sessionMs > 0)
            out << "  GPU Busy:             " << std::fixed << std::setprecision(1)
                << (totalMs / sessionMs * 100) << "%\n";
    }

    out << "  Avg Duration:         " << fmtDuration(totalMs / kernels_.size()) << "\n";
    out << "  Median Duration:      " << fmtDuration(durations[durations.size() / 2]) << "\n";
    out << "  Min Duration:         " << fmtDuration(*minIt) << "\n";
    out << "  Max Duration:         " << fmtDuration(*maxIt) << "\n";
}

void TextReport::writeTopKernels(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Top " << top_n_ << " Kernels by Total GPU Time\n" << SEP << "\n";
    if (kernels_.empty()) { out << "  (No kernel data)\n"; return; }

    // Aggregate per kernel name using AggStats
    std::map<std::string, AggStats> grouped;
    for (const auto& k : kernels_)
        grouped[k.name].add(k.duration_ms);

    auto ranked = sortedTopN(grouped, top_n_, [](const AggStats& s) { return s.total; });

    out << "  " << std::left << std::setw(4) << "#"
        << std::setw(40) << "Kernel"
        << std::right << std::setw(6) << "Calls"
        << std::setw(12) << "Total" << std::setw(12) << "Avg" << std::setw(12) << "Max" << "\n";
    out << "  " << std::string(86, '-') << "\n";

    int rank = 0;
    for (const auto& [name, st] : ranked) {
        out << "  " << std::left << std::setw(4) << ++rank
            << std::setw(40) << truncate(shortenKernelName(name), 38)
            << std::right << std::setw(6) << st.count
            << std::setw(12) << fmtDuration(st.total)
            << std::setw(12) << fmtDuration(st.avg())
            << std::setw(12) << fmtDuration(st.max_val) << "\n";
    }
}

void TextReport::writeKernelDetails(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Kernel Details (Top " << top_n_ << ")\n" << SEP << "\n";

    bool hasDetails = std::any_of(kernels_.begin(), kernels_.end(),
                                  [](const KernelRecord& k) { return k.occupancy >= 0; });
    if (!hasDetails || kernels_.empty()) { out << "  (No kernel detail data)\n"; return; }

    // Rank by total GPU time
    std::map<std::string, double> totalByName;
    for (const auto& k : kernels_) totalByName[k.name] += k.duration_ms;
    auto ranked = sortedTopN(totalByName, top_n_, [](double v) { return v; });

    for (const auto& [name, _] : ranked) {
        // Find representative kernel with detail data
        auto rep = std::find_if(kernels_.begin(), kernels_.end(),
                                [&](const KernelRecord& k) { return k.name == name && k.occupancy >= 0; });
        if (rep == kernels_.end()) continue;

        std::string shortName = shortenKernelName(name);
        out << "\n  " << shortName << "\n  " << std::string(shortName.size(), '=') << "\n";
        out << "    Grid:               " << rep->grid << "\n";
        out << "    Block:              " << rep->block << "\n";

        if (rep->occupancy >= 0)
            out << "    Occupancy:          " << std::fixed << std::setprecision(1)
                << (rep->occupancy * 100) << "%\n";

        auto writeOcc = [&](const char* label, float val) {
            if (val >= 0)
                out << "    " << std::left << std::setw(20) << (std::string(label) + ":")
                    << std::fixed << std::setprecision(1) << (val * 100) << "%\n";
        };
        writeOcc("Reg Occupancy", rep->reg_occupancy);
        writeOcc("SMem Occupancy", rep->smem_occupancy);
        writeOcc("Warp Occupancy", rep->warp_occupancy);
        writeOcc("Block Occupancy", rep->block_occupancy);

        if (!rep->limiting_resource.empty())
            out << "    Limiting Resource:  " << rep->limiting_resource << "\n";
        if (rep->num_regs > 0)
            out << "    Registers/Thread:   " << rep->num_regs << "\n";
        out << "    Shared Memory:      " << fmtBytes(rep->dyn_shared)
            << " dyn + " << fmtBytes(rep->static_shared) << " static\n";
    }
}

void TextReport::writeMemcpySummary(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Memory Transfer Summary\n" << SEP << "\n";
    if (memcpy_.empty()) { out << "  (No memory transfer data)\n"; return; }

    uint64_t totalBytes = std::accumulate(memcpy_.begin(), memcpy_.end(), uint64_t(0),
                                          [](uint64_t sum, const MemcpyRecord& m) { return sum + m.bytes; });

    out << "  Total Transfers:      " << memcpy_.size() << "\n";
    out << "  Total Bytes:          " << fmtBytes(totalBytes) << "\n\n";

    // Group by copy_kind
    std::map<int, std::vector<const MemcpyRecord*>> grouped;
    for (const auto& m : memcpy_) grouped[m.copy_kind].push_back(&m);

    out << "  " << std::left << std::setw(12) << "Direction"
        << std::right << std::setw(8) << "Count"
        << std::setw(16) << "Total Bytes" << std::setw(18) << "Avg Throughput" << "\n";
    out << "  " << std::string(54, '-') << "\n";

    for (const auto& [kind, recs] : grouped) {
        uint64_t kb = std::accumulate(recs.begin(), recs.end(), uint64_t(0),
                                      [](uint64_t s, const MemcpyRecord* r) { return s + r->bytes; });
        double totalDurNs = std::accumulate(recs.begin(), recs.end(), 0.0,
                                            [](double s, const MemcpyRecord* r) { return s + r->duration_ms * 1e6; });
        std::string tp;
        if (totalDurNs > 0) {
            std::ostringstream toss;
            toss << std::fixed << std::setprecision(2) << (static_cast<double>(kb) / totalDurNs) << " GB/s";
            tp = toss.str();
        }
        out << "  " << std::left << std::setw(12) << resolveCopyKind(kind)
            << std::right << std::setw(8) << recs.size()
            << std::setw(16) << fmtBytes(kb) << std::setw(18) << tp << "\n";
    }
}

void TextReport::writeSystemMetrics(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  System Metrics\n" << SEP << "\n";
    if (device_metrics_.empty() && host_metrics_.empty()) {
        out << "  (No system metric data)\n"; return;
    }

    if (!device_metrics_.empty()) {
        out << "  GPU Metrics:\n";

        // Functional aggregation with accumulate
        struct GpuAgg {
            double sumUtil=0, maxUtil=0, minUtil=1e9;
            double sumTemp=0, maxTemp=0;
            double sumJTemp=0, maxJTemp=0; int jTempN=0;
            double sumMTemp=0, maxMTemp=0; int mTempN=0;
            double sumPow=0, maxPow=0;
            double sumVolt=0, maxVolt=0; int voltN=0;
            double sumFan=0, maxFan=0; int fanN=0;
            double sumClk=0, peakClk=0; int clkN=0;
            double sumMemClk=0, peakMemClk=0; int memClkN=0;
            double sumPcie=0, maxPcie=0; int pcieN=0;
            uint64_t maxMem=0;
            uint64_t lastEnergy=0, firstEnergy=0; bool hasEnergy=false;
            uint64_t maxEccCorr=0, maxEccUncorr=0;
            int n=0;
        };
        auto agg = std::accumulate(device_metrics_.begin(), device_metrics_.end(), GpuAgg{},
            [](GpuAgg a, const DeviceMetricRecord& m) {
                a.sumUtil += m.gpu_util;
                a.maxUtil = (std::max)(a.maxUtil, (double)m.gpu_util);
                a.minUtil = (std::min)(a.minUtil, (double)m.gpu_util);
                a.sumTemp += m.temp_c;
                a.maxTemp = (std::max)(a.maxTemp, (double)m.temp_c);
                a.sumPow  += m.power_mw;
                a.maxPow  = (std::max)(a.maxPow, (double)m.power_mw);
                a.maxMem  = (std::max)(a.maxMem, m.used_mib);
                if (m.clock_sm > 0) {
                    a.sumClk += m.clock_sm; a.peakClk = (std::max)(a.peakClk, (double)m.clock_sm); a.clkN++;
                }
                if (m.temp_junction_c > 0) {
                    a.sumJTemp += m.temp_junction_c; a.maxJTemp = (std::max)(a.maxJTemp, (double)m.temp_junction_c); a.jTempN++;
                }
                if (m.temp_mem_c > 0) {
                    a.sumMTemp += m.temp_mem_c; a.maxMTemp = (std::max)(a.maxMTemp, (double)m.temp_mem_c); a.mTempN++;
                }
                if (m.voltage_mv > 0) {
                    a.sumVolt += m.voltage_mv; a.maxVolt = (std::max)(a.maxVolt, (double)m.voltage_mv); a.voltN++;
                }
                if (m.fan_speed_pct > 0) {
                    a.sumFan += m.fan_speed_pct; a.maxFan = (std::max)(a.maxFan, (double)m.fan_speed_pct); a.fanN++;
                }
                if (m.clock_mem > 0) {
                    a.sumMemClk += m.clock_mem; a.peakMemClk = (std::max)(a.peakMemClk, (double)m.clock_mem); a.memClkN++;
                }
                if (m.pcie_bw_bps > 0) {
                    a.sumPcie += m.pcie_bw_bps; a.maxPcie = (std::max)(a.maxPcie, (double)m.pcie_bw_bps); a.pcieN++;
                }
                if (m.energy_uj > 0) {
                    if (!a.hasEnergy) { a.firstEnergy = m.energy_uj; a.hasEnergy = true; }
                    a.lastEnergy = m.energy_uj;
                }
                a.maxEccCorr   = (std::max)(a.maxEccCorr, m.ecc_corrected);
                a.maxEccUncorr = (std::max)(a.maxEccUncorr, m.ecc_uncorrected);
                a.n++; return a;
            });

        out << "    Utilization:        avg " << std::fixed << std::setprecision(1) << (agg.sumUtil / agg.n)
            << "%  peak " << std::setprecision(0) << agg.maxUtil << "%  min " << agg.minUtil << "%\n";
        out << "    Temperature:        avg " << std::setprecision(1) << (agg.sumTemp / agg.n)
            << " C  peak " << std::setprecision(0) << agg.maxTemp << " C\n";
        if (agg.jTempN > 0)
            out << "    Junction Temp:      avg " << std::setprecision(1) << (agg.sumJTemp / agg.jTempN)
                << " C  peak " << std::setprecision(0) << agg.maxJTemp << " C\n";
        if (agg.mTempN > 0)
            out << "    Memory Temp:        avg " << std::setprecision(1) << (agg.sumMTemp / agg.mTempN)
                << " C  peak " << std::setprecision(0) << agg.maxMTemp << " C\n";
        out << "    Power:              avg " << fmtPower(agg.sumPow / agg.n) << "  peak " << fmtPower(agg.maxPow) << "\n";
        if (agg.voltN > 0)
            out << "    Voltage:            avg " << std::setprecision(0) << (agg.sumVolt / agg.voltN)
                << " mV  peak " << agg.maxVolt << " mV\n";
        if (agg.hasEnergy && agg.lastEnergy > agg.firstEnergy) {
            double energyJ = (agg.lastEnergy - agg.firstEnergy) / 1e6;
            if (energyJ >= 1000.0)
                out << "    Energy:             " << std::setprecision(2) << (energyJ / 1000.0) << " kJ\n";
            else
                out << "    Energy:             " << std::setprecision(2) << energyJ << " J\n";
        }
        if (agg.fanN > 0)
            out << "    Fan Speed:          avg " << std::setprecision(0) << (agg.sumFan / agg.fanN)
                << "%  peak " << agg.maxFan << "%\n";
        out << "    VRAM Usage:         peak " << agg.maxMem << " MiB\n";
        if (agg.clkN > 0)
            out << "    SM Clock:           avg " << std::setprecision(0) << (agg.sumClk / agg.clkN)
                << " MHz  peak " << agg.peakClk << " MHz\n";
        if (agg.memClkN > 0)
            out << "    Memory Clock:       avg " << std::setprecision(0) << (agg.sumMemClk / agg.memClkN)
                << " MHz  peak " << agg.peakMemClk << " MHz\n";
        if (agg.pcieN > 0) {
            double avgGbps = (agg.sumPcie / agg.pcieN) / 1e9;
            double peakGbps = agg.maxPcie / 1e9;
            out << "    PCIe Bandwidth:     avg " << std::setprecision(1) << avgGbps
                << " GB/s  peak " << peakGbps << " GB/s\n";
        }
        if (agg.maxEccCorr > 0 || agg.maxEccUncorr > 0)
            out << "    ECC Errors:         " << agg.maxEccCorr << " corrected, "
                << agg.maxEccUncorr << " uncorrected\n";
    }

    if (!host_metrics_.empty()) {
        if (!device_metrics_.empty()) out << "\n";
        out << "  Host Metrics:\n";

        struct HostAgg { double sumCpu=0, maxCpu=0; uint64_t maxRam=0, totalRam=0; int n=0; };
        auto agg = std::accumulate(host_metrics_.begin(), host_metrics_.end(), HostAgg{},
            [](HostAgg a, const HostMetricRecord& m) {
                a.sumCpu += m.cpu_pct; a.maxCpu = (std::max)(a.maxCpu, m.cpu_pct);
                a.maxRam = (std::max)(a.maxRam, m.ram_used_mib);
                if (m.ram_total_mib > 0) a.totalRam = m.ram_total_mib;
                a.n++; return a;
            });

        out << "    CPU Utilization:    avg " << std::fixed << std::setprecision(1)
            << (agg.sumCpu / agg.n) << "%  peak " << agg.maxCpu << "%\n";
        if (agg.totalRam > 0)
            out << "    RAM Usage:          peak " << agg.maxRam << " / " << agg.totalRam
                << " MiB (" << std::setprecision(1) << (agg.maxRam * 100.0 / agg.totalRam) << "%)\n";
    }
}

void TextReport::writeScopeSummary(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Scope Summary\n" << SEP << "\n";

    bool hasKernelScopes = std::any_of(kernels_.begin(), kernels_.end(),
                                       [](const KernelRecord& k) { return !k.user_scope.empty(); });
    if (scope_events_.empty() && !hasKernelScopes) { out << "  (No scope data)\n"; return; }

    // Scope event timing (begin/end pairs)
    if (!scope_events_.empty()) {
        std::unordered_map<uint64_t, int64_t> beginTimes;
        std::unordered_map<uint64_t, std::string> beginNames;
        std::map<std::string, AggStats> stats;

        for (const auto& se : scope_events_) {
            if (se.event_type == 0) {
                beginTimes[se.scope_instance_id] = se.ts_ns;
                beginNames[se.scope_instance_id] = se.name;
            } else if (se.event_type == 1) {
                auto it = beginTimes.find(se.scope_instance_id);
                if (it != beginTimes.end())
                    stats[beginNames[se.scope_instance_id]].add((se.ts_ns - it->second) / 1e6);
            }
        }

        if (!stats.empty()) {
            auto ranked = sortedTopN(stats, 0, [](const AggStats& s) { return s.total; });
            out << "  Scope Timing:\n";
            out << "  " << std::left << std::setw(30) << "Scope"
                << std::right << std::setw(6) << "Calls"
                << std::setw(12) << "Total" << std::setw(12) << "Avg" << std::setw(12) << "Max" << "\n";
            out << "  " << std::string(72, '-') << "\n";
            for (const auto& [name, st] : ranked)
                out << "  " << std::left << std::setw(30) << truncate(name, 28)
                    << std::right << std::setw(6) << st.count
                    << std::setw(12) << fmtDuration(st.total)
                    << std::setw(12) << fmtDuration(st.avg())
                    << std::setw(12) << fmtDuration(st.max_val) << "\n";
        }
    }

    // GPU time by scope (from kernel user_scope)
    if (hasKernelScopes) {
        std::map<std::string, AggStats> scopeGpu;
        for (const auto& k : kernels_) {
            if (k.user_scope.empty()) continue;
            auto pipe = k.user_scope.find('|');
            scopeGpu[(pipe != std::string::npos) ? k.user_scope.substr(0, pipe) : k.user_scope]
                .add(k.duration_ms);
        }

        if (!scopeGpu.empty()) {
            auto ranked = sortedTopN(scopeGpu, 0, [](const AggStats& s) { return s.total; });
            out << "\n  GPU Time by Scope:\n";
            out << "  " << std::left << std::setw(30) << "Scope"
                << std::right << std::setw(8) << "Kernels"
                << std::setw(14) << "GPU Time" << std::setw(12) << "Avg" << "\n";
            out << "  " << std::string(64, '-') << "\n";
            for (const auto& [name, sg] : ranked)
                out << "  " << std::left << std::setw(30) << truncate(name, 28)
                    << std::right << std::setw(8) << sg.count
                    << std::setw(14) << fmtDuration(sg.total)
                    << std::setw(12) << fmtDuration(sg.avg()) << "\n";
        }
    }
}

// ── Profile analysis helpers ────────────────────────────────────────────────

static std::string fmtCount(uint64_t v) {
    std::string s = std::to_string(v);
    int n = static_cast<int>(s.size());
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) out += ',';
        out += s[i];
    }
    return out;
}

static std::string makeBar(double pct, int maxWidth = 20) {
    int filled = static_cast<int>(pct / 100.0 * maxWidth + 0.5);
    if (filled < 0) filled = 0;
    if (filled > maxWidth) filled = maxWidth;
    return std::string(filled, '#');
}

// FuncProfile is defined in hint_engine.hpp
using FuncProfile = gpufl::report::FuncProfile;

void TextReport::writeProfileAnalysis(std::ostringstream& out) const {
    out << "\n" << SEP << "\n  Profile / SASS Analysis\n" << SEP << "\n";
    if (profile_samples_.empty()) { out << "  (No profile sample data)\n"; return; }

    // Convert a raw CUPTI stall metric name to a human-readable short name.
    // e.g. "smsp__pcsamp_warps_issue_stalled_wait_not_issued" → "Wait (not issued)"
    auto shortenStallName = [](const std::string& raw) -> std::string {
        const std::string prefix = "smsp__pcsamp_warps_issue_stalled_";
        std::string s = raw;
        if (s.size() > prefix.size() && s.substr(0, prefix.size()) == prefix)
            s = s.substr(prefix.size());

        // Handle "_not_issued" suffix
        const std::string notIssued = "_not_issued";
        bool isNotIssued = false;
        if (s.size() > notIssued.size() &&
            s.substr(s.size() - notIssued.size()) == notIssued) {
            s = s.substr(0, s.size() - notIssued.size());
            isNotIssued = true;
        }

        // Replace underscores with spaces and capitalize first letter of each word
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '_') s[i] = ' ';
            if (i == 0 || (i > 0 && s[i-1] == ' '))
                s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
        }

        if (isNotIssued) s += " (idle)";
        return s;
    };

    // Resolve stall reason display name: prefer reason_name from log, then
    // metric_name (for PC sampling, reason_name is interned as metric_name),
    // then fall back to the static map. Always shorten CUPTI-style names.
    auto resolveStallDisplay = [&](const ProfileSampleRecord& ps) -> std::string {
        std::string raw;
        if (!ps.reason_name.empty())
            raw = ps.reason_name;
        else if (ps.sample_kind == 0 && !ps.metric_name.empty())
            raw = ps.metric_name;
        else
            return resolveStallReason(ps.stall_reason);
        return shortenStallName(raw);
    };

    // ── Collect per-function data ───────────────────────────────────────────
    std::map<std::string, FuncProfile> byFunc;
    for (const auto& ps : profile_samples_) {
        std::string fn = ps.function_name.empty() ? "(unknown)" : ps.function_name;
        auto& fp = byFunc[fn];

        if (ps.stall_reason > 1) {
            std::string reason = resolveStallDisplay(ps);
            fp.stalls[reason] += ps.metric_value;
            fp.totalStalls += ps.metric_value;
        }
        if (ps.metric_name == "smsp__sass_inst_executed")
            fp.warpInsts += ps.metric_value;
        else if (ps.metric_name == "smsp__sass_thread_inst_executed")
            fp.threadInsts += ps.metric_value;
        else if (ps.metric_name == "smsp__sass_sectors_mem_global")
            fp.globalSectors += ps.metric_value;
        else if (ps.metric_name == "smsp__sass_sectors_mem_global_ideal")
            fp.idealSectors += ps.metric_value;
        else if (ps.metric_name == "smsp__sass_sectors_mem_global_op_ld_ideal")
            fp.idealLoadSectors += ps.metric_value;
        else if (ps.metric_name == "smsp__sass_sectors_mem_global_op_st_ideal")
            fp.idealStoreSectors += ps.metric_value;
    }

    // ── Rank functions by total stall samples ───────────────────────────────
    std::vector<std::pair<std::string, const FuncProfile*>> ranked;
    for (const auto& [fn, fp] : byFunc)
        ranked.emplace_back(fn, &fp);
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second->totalStalls > b.second->totalStalls; });
    if (static_cast<int>(ranked.size()) > top_n_)
        ranked.resize(top_n_);

    // ── Write per-function analysis ─────────────────────────────────────────
    for (const auto& [fn, fp] : ranked) {
        std::string shortName = shortenKernelName(fn);
        out << "\n  " << shortName;
        if (fp->totalStalls > 0)
            out << "  (" << fmtCount(fp->totalStalls) << " stall samples)";
        out << "\n  " << std::string((std::min)(shortName.size() + 30, size_t(72)), '-') << "\n";

        // Stall distribution
        if (!fp->stalls.empty()) {
            auto stallRanked = sortedTopN(fp->stalls, 0, [](uint64_t v) { return static_cast<double>(v); });
            out << "    Stalls:\n";
            for (const auto& [reason, count] : stallRanked) {
                double pct = fp->totalStalls > 0 ? count * 100.0 / fp->totalStalls : 0;
                out << "      " << std::left << std::setw(28) << truncate(reason, 26)
                    << std::right << std::setw(8) << fmtCount(count)
                    << std::setw(7) << std::fixed << std::setprecision(1) << pct << "%  "
                    << makeBar(pct) << "\n";
            }
        }

        // Instruction analysis
        if (fp->warpInsts > 0 || fp->threadInsts > 0) {
            out << "    Instructions:\n";
            if (fp->warpInsts > 0)
                out << "      Warp Insts:           " << std::setw(16) << fmtCount(fp->warpInsts) << "\n";
            if (fp->threadInsts > 0)
                out << "      Thread Insts:         " << std::setw(16) << fmtCount(fp->threadInsts) << "\n";
            if (fp->warpInsts > 0 && fp->threadInsts > 0) {
                double ratio = static_cast<double>(fp->threadInsts) / fp->warpInsts;
                double eff = ratio / 32.0 * 100;
                out << "      Warp Efficiency:      " << std::setw(15)
                    << (std::to_string(static_cast<int>(ratio * 10) / 10) + "." +
                        std::to_string(static_cast<int>(ratio * 10) % 10))
                    << " / 32 (" << std::fixed << std::setprecision(1) << eff << "%)\n";
            }
        }

        // Memory analysis
        if (fp->globalSectors > 0) {
            const uint64_t effectiveIdeal = fp->idealSectors > 0
                ? fp->idealSectors
                : fp->idealLoadSectors + fp->idealStoreSectors;
            out << "    Memory:\n";
            out << "      Global Sectors:       " << std::setw(16) << fmtCount(fp->globalSectors) << "\n";
            if (effectiveIdeal > 0) {
                out << "      Ideal Sectors:        " << std::setw(16) << fmtCount(effectiveIdeal);
                if (fp->idealSectors == 0) out << " (ld+st)";
                out << "\n";
                double memEff = static_cast<double>(effectiveIdeal) / fp->globalSectors * 100;
                out << "      Memory Efficiency:    " << std::setw(15) << std::fixed
                    << std::setprecision(1) << memEff << "%\n";
            } else {
                out << "      Memory Efficiency:    (not available on this GPU)\n";
            }
        }

        // Interpretation hints
        {
            auto hints = computeHints(*fp);
            if (!hints.empty()) {
                out << "    Hints:\n";
                for (const auto& h : hints)
                    out << "      * " << h << "\n";
            }
        }
    }

    // ── Global SASS metrics (other metrics not covered above) ───────────────
    std::map<std::string, uint64_t> otherMetrics;
    for (const auto& ps : profile_samples_) {
        if (ps.metric_name.empty()) continue;
        if (ps.stall_reason > 1 && ps.sample_kind == 0) continue;  // stall data already shown
        if (ps.metric_name == "smsp__sass_inst_executed") continue;
        if (ps.metric_name == "smsp__sass_thread_inst_executed") continue;
        if (ps.metric_name == "smsp__sass_sectors_mem_global") continue;
        if (ps.metric_name == "smsp__sass_sectors_mem_global_ideal") continue;
        if (ps.metric_name == "smsp__sass_sectors_mem_global_op_ld_ideal") continue;
        if (ps.metric_name == "smsp__sass_sectors_mem_global_op_st_ideal") continue;
        otherMetrics[ps.metric_name] += ps.metric_value;
    }

    if (!otherMetrics.empty()) {
        auto metricRanked = sortedTopN(otherMetrics, 0, [](uint64_t v) { return static_cast<double>(v); });
        out << "\n  Other SASS Metrics:\n";
        out << "  " << std::left << std::setw(50) << "Metric"
            << std::right << std::setw(16) << "Total" << "\n";
        out << "  " << std::string(66, '-') << "\n";
        for (const auto& [metric, total] : metricRanked)
            out << "  " << std::left << std::setw(50) << truncate(metric, 48)
                << std::right << std::setw(16) << fmtCount(total) << "\n";
    }
}

}  // namespace report
}  // namespace gpufl
