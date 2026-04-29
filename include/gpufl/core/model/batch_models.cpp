#include "gpufl/core/model/batch_models.hpp"

#include <sstream>

#include "gpufl/core/host_info.hpp"
#include "gpufl/core/model/model_utils.hpp"

namespace gpufl::model {

// ── KernelEventBatchModel ─────────────────────────────────────────────────

std::string KernelEventBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().start_ns;

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"kernel_event_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"kernel_id\",\"stream_id\","
           "\"duration_ns\",\"corr_id\",\"dyn_shared\",\"num_regs\",\"has_"
           "details\"]"
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.start_ns - base) << ',' << r.kernel_id << ','
            << r.stream_id << ',' << r.duration_ns << ',' << r.corr_id << ','
            << r.dyn_shared << ',' << r.num_regs << ','
            << static_cast<int>(r.has_details) << ']';
    }
    oss << "]}";
    return oss.str();
}

// ── KernelDetailModel ─────────────────────────────────────────────────────

std::string KernelDetailModel::buildJson() const {
    std::ostringstream oss;
    oss << std::fixed;
    oss << "{\"version\":1,\"type\":\"kernel_detail\""
        << ",\"session_id\":\"" << jsonEscape(r_.session_id) << '"'
        << ",\"pid\":" << r_.pid << ",\"app\":\"" << jsonEscape(r_.app) << '"'
        << ",\"corr_id\":" << r_.corr_id << ",\"grid\":\"(" << r_.grid_x << ','
        << r_.grid_y << ',' << r_.grid_z << ")\""
        << ",\"block\":\"(" << r_.block_x << ',' << r_.block_y << ','
        << r_.block_z << ")\""
        << ",\"static_shared\":" << r_.static_shared
        << ",\"local_bytes\":" << r_.local_bytes
        << ",\"const_bytes\":" << r_.const_bytes
        << ",\"occupancy\":" << r_.occupancy
        << ",\"reg_occupancy\":" << r_.reg_occupancy
        << ",\"smem_occupancy\":" << r_.smem_occupancy
        << ",\"warp_occupancy\":" << r_.warp_occupancy
        << ",\"block_occupancy\":" << r_.block_occupancy
        << ",\"limiting_resource\":\"" << r_.limiting_resource << '"'
        << ",\"max_active_blocks\":" << r_.max_active_blocks
        << ",\"local_mem_total_bytes\":" << r_.local_mem_total
        << ",\"local_mem_per_thread_bytes\":" << r_.local_mem_per_thread
        << ",\"cache_config_requested\":"
        << static_cast<int>(r_.cache_config_requested)
        << ",\"cache_config_executed\":"
        << static_cast<int>(r_.cache_config_executed)
        << ",\"shared_mem_executed_bytes\":" << r_.shared_mem_executed
        << ",\"user_scope\":\"" << jsonEscape(r_.user_scope) << '"'
        << ",\"stack_trace\":\"" << jsonEscape(r_.stack_trace) << '"' << '}';
    return oss.str();
}

// ── MemcpyEventBatchModel ─────────────────────────────────────────────────

std::string MemcpyEventBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().start_ns;

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"memcpy_event_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"stream_id\",\"duration_ns\","
           "\"bytes\",\"copy_kind\",\"corr_id\"]"
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.start_ns - base) << ',' << r.stream_id << ','
            << r.duration_ns << ',' << r.bytes << ',' << r.copy_kind << ','
            << r.corr_id << ']';
    }
    oss << "]}";
    return oss.str();
}

// ── DeviceMetricBatchModel ────────────────────────────────────────────────

std::string DeviceMetricBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().ts_ns;
    const bool has_extended_metrics = [&rows]() {
        for (const auto& r : rows) {
            if (r.fan_speed_pct != 0 || r.temp_mem_c != 0 ||
                r.temp_junction_c != 0 || r.voltage_mv != 0 ||
                r.energy_uj != 0 || r.clock_mem != 0 ||
                r.pcie_bw_bps != 0 || r.ecc_corrected != 0 ||
                r.ecc_uncorrected != 0) {
                return true;
            }
        }
        return false;
    }();

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"device_metric_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"device_id\",\"gpu_util\","
           "\"mem_util\",\"temp_c\",\"power_mw\",\"used_mib\",\"total_mib\",\"clock_sm\"";
    if (has_extended_metrics) {
        oss << ",\"fan_speed_pct\",\"temp_mem_c\",\"temp_junction_c\","
               "\"voltage_mv\",\"energy_uj\",\"clock_mem\","
               "\"pcie_bw_bps\",\"ecc_corrected\",\"ecc_uncorrected\"";
    }
    oss << ']'
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.ts_ns - base) << ',' << r.device_id << ','
            << r.gpu_util << ',' << r.mem_util << ',' << r.temp_c << ','
            << r.power_mw << ',' << r.used_mib << ',' << r.total_mib << ',' << r.clock_sm;
        if (has_extended_metrics) {
            oss << ',' << r.fan_speed_pct << ',' << r.temp_mem_c << ','
                << r.temp_junction_c << ',' << r.voltage_mv << ','
                << r.energy_uj << ',' << r.clock_mem << ','
                << r.pcie_bw_bps << ',' << r.ecc_corrected << ','
                << r.ecc_uncorrected;
        }
        oss << ']';
    }
    oss << "]}";
    return oss.str();
}

// ── ScopeEventBatchModel ──────────────────────────────────────────────────

std::string ScopeEventBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().ts_ns;

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"scope_event_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"scope_instance_id\",\"name_id\","
           "\"event_type\",\"depth\"]"
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.ts_ns - base) << ',' << r.scope_instance_id << ','
            << r.name_id << ',' << static_cast<int>(r.event_type) << ','
            << r.depth << ']';
    }
    oss << "]}";
    return oss.str();
}

// ── ProfileSampleBatchModel ───────────────────────────────────────────────

std::string ProfileSampleBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().ts_ns;

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"profile_sample_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"corr_id\",\"device_id\",\"function_id\","
           "\"pc_offset\",\"metric_id\",\"metric_value\",\"stall_reason\","
           "\"sample_kind\",\"scope_name_id\","
           "\"source_file_id\",\"source_line\"]"
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.ts_ns - base) << ',' << r.corr_id << ',' << r.device_id
            << ',' << r.function_id << ',' << r.pc_offset << ',' << r.metric_id
            << ',' << r.metric_value << ',' << r.stall_reason << ','
            << static_cast<int>(r.sample_kind) << ',' << r.scope_name_id << ','
            << r.source_file_id << ',' << r.source_line << ']';
    }
    oss << "]}";
    return oss.str();
}

// ── HostMetricBatchModel ──────────────────────────────────────────────────

std::string HostMetricBatchModel::buildJson() const {
    const auto& rows = buf_.rows();
    if (rows.empty()) return {};
    const int64_t base = rows.front().ts_ns;

    // Hostname / ip_addr are added to the batch ENVELOPE (not per-row)
    // so the backend can copy them onto every host_metric record at
    // ingestion time. A file-tailing agent reading this NDJSON gets
    // the host label without needing to call gethostname() itself —
    // important when the agent runs on a different machine than the
    // workload (sidecar / centralized collector). Per-row replication
    // would waste bytes; the value is constant for a session.
    const std::string hostname = gpufl::getLocalHostname();
    const std::string ipAddr   = gpufl::getLocalIpAddr();

    std::ostringstream oss;
    oss << "{\"version\":1,\"type\":\"host_metric_batch\""
        << ",\"session_id\":\"" << jsonEscape(session_id_) << '"'
        << ",\"hostname\":\""   << jsonEscape(hostname)    << '"'
        << ",\"ip_addr\":\""    << jsonEscape(ipAddr)      << '"'
        << ",\"batch_id\":" << batch_id_ << ",\"base_time_ns\":" << base
        << ",\"columns\":[\"dt_ns\",\"cpu_pct_x100\",\"ram_used_mib\",\"ram_"
           "total_mib\"]"
        << ",\"rows\":[";

    bool first = true;
    for (const auto& r : rows) {
        if (!first) oss << ',';
        first = false;
        oss << '[' << (r.ts_ns - base) << ',' << r.cpu_pct_x100 << ','
            << r.ram_used_mib << ',' << r.ram_total_mib << ']';
    }
    oss << "]}";
    return oss.str();
}

}  // namespace gpufl::model
