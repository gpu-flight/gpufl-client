#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpufl/report/json_reader.hpp"

namespace gpufl {
namespace report {

class TextReport {
   public:
    struct Options {
        std::string log_dir;
        std::string log_prefix;
        int top_n = 10;
    };

    explicit TextReport(const Options& opts);

    std::string generate() const;

   private:
    // ── Data records ────────────────────────────────────────────────────────

    struct KernelRecord {
        std::string name;
        std::string session_id;
        int64_t start_ns = 0;
        int64_t end_ns = 0;
        double duration_ms = 0;
        uint32_t stream_id = 0;
        unsigned corr_id = 0;
        int num_regs = 0;
        int dyn_shared = 0;
        bool has_details = false;
        std::string grid;
        std::string block;
        float occupancy = -1;
        float reg_occupancy = -1;
        float smem_occupancy = -1;
        float warp_occupancy = -1;
        float block_occupancy = -1;
        std::string limiting_resource;
        int static_shared = 0;
        std::string user_scope;
    };

    struct MemcpyRecord {
        int64_t start_ns = 0;
        double duration_ms = 0;
        uint64_t bytes = 0;
        int copy_kind = 0;
    };

    struct DeviceMetricRecord {
        int64_t ts_ns = 0;
        int gpu_util = 0;
        int mem_util = 0;
        int temp_c = 0;
        int power_mw = 0;
        uint64_t used_mib = 0;
        int clock_sm = 0;
    };

    struct HostMetricRecord {
        int64_t ts_ns = 0;
        double cpu_pct = 0;
        uint64_t ram_used_mib = 0;
        uint64_t ram_total_mib = 0;
    };

    struct ScopeEventRecord {
        int64_t ts_ns = 0;
        uint64_t scope_instance_id = 0;
        std::string name;
        int event_type = 0;
    };

    struct ProfileSampleRecord {
        std::string function_name;
        std::string metric_name;
        uint64_t metric_value = 0;
        int stall_reason = 0;
        int sample_kind = 0;
    };

    struct SessionInfo {
        std::string app_name;
        std::string session_id;
        int64_t start_ns = 0;
        int64_t end_ns = 0;
        std::string gpu_name;
        int compute_major = 0;
        int compute_minor = 0;
        int sm_count = 0;
        int shared_mem_per_block = 0;
        int regs_per_block = 0;
        int l2_cache_size = 0;
    };

    // Aggregation helper used by multiple sections
    struct AggStats {
        int count = 0;
        double total = 0;
        double max_val = 0;
        void add(double v) { count++; total += v; max_val = (std::max)(max_val, v); }
        double avg() const { return count > 0 ? total / count : 0; }
    };

    // ── Parsed state ────────────────────────────────────────────────────────

    SessionInfo info_;
    std::vector<KernelRecord> kernels_;
    std::vector<MemcpyRecord> memcpy_;
    std::vector<DeviceMetricRecord> device_metrics_;
    std::vector<HostMetricRecord> host_metrics_;
    std::vector<ScopeEventRecord> scope_events_;
    std::vector<ProfileSampleRecord> profile_samples_;
    int top_n_;

    // ── Parsing (constructor helpers) ────────────────────────────────────────

    void parseLogFiles(const Options& opts);
    void parseDictionaries(const std::vector<JsonValue>& records,
                           std::unordered_map<int, std::string>& kernel_dict,
                           std::unordered_map<int, std::string>& scope_name_dict,
                           std::unordered_map<int, std::string>& function_dict,
                           std::unordered_map<int, std::string>& metric_dict);
    void parseDeviceLog(const std::vector<JsonValue>& records,
                        const std::unordered_map<int, std::string>& kernel_dict);
    void parseScopeLog(const std::vector<JsonValue>& records,
                       const std::unordered_map<int, std::string>& scope_name_dict,
                       const std::unordered_map<int, std::string>& function_dict,
                       const std::unordered_map<int, std::string>& metric_dict);
    void parseSystemLog(const std::vector<JsonValue>& records);
    void mergeKernelDetails(std::unordered_map<unsigned, JsonValue>& details);

    static void parseJobStart(const JsonValue& record, SessionInfo& info);

    // ── Section writers ─────────────────────────────────────────────────────

    void writeHeader(std::ostringstream& out) const;
    void writeSessionSummary(std::ostringstream& out) const;
    void writeKernelSummary(std::ostringstream& out) const;
    void writeTopKernels(std::ostringstream& out) const;
    void writeKernelDetails(std::ostringstream& out) const;
    void writeMemcpySummary(std::ostringstream& out) const;
    void writeSystemMetrics(std::ostringstream& out) const;
    void writeScopeSummary(std::ostringstream& out) const;
    void writeProfileAnalysis(std::ostringstream& out) const;
};

}  // namespace report
}  // namespace gpufl
