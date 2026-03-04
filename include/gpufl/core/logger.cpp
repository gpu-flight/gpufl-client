#include "gpufl/core/logger.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {
namespace fs = std::filesystem;

static inline std::string jsonEscape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u" << std::hex << (int)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}
static std::string cudaStaticDeviceToJson(
    const std::vector<CudaStaticDeviceInfo>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (auto& dev : devs) {
        if (!first) oss << ",";
        first = false;
        oss << "{" << "\"id\":" << dev.id << ",\"name\":\""
            << jsonEscape(dev.name) << "\"" << ",\"uuid\":\""
            << jsonEscape(dev.uuid) << "\"" << ",\"compute_major\":\""
            << dev.compute_major << "\""
            << ",\"compute_minor\":" << dev.compute_minor
            << ",\"l2_cache_size\":" << dev.l2_cache_size
            << ",\"shared_mem_per_block\":" << dev.shared_mem_per_block
            << ",\"regs_per_block\":" << dev.regs_per_block
            << ",\"multi_processor_count\":" << dev.multi_processor_count
            << ",\"warp_size\":" << dev.warp_size << "}";
    }
    oss << "]";
    return oss.str();
}

static std::string devicesToJson(const std::vector<DeviceSample>& devs) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (auto& dev : devs) {
        if (!first) oss << ",";
        first = false;
        oss << "{" << "\"id\":" << dev.device_id << ",\"name\":\""
            << jsonEscape(dev.name) << "\"" << ",\"uuid\":\""
            << jsonEscape(dev.uuid) << "\"" << ",\"vendor\":\""
            << jsonEscape(dev.vendor) << "\"" << ",\"pci_bus\":" << dev.pci_bus_id
            << ",\"used_mib\":" << dev.used_mib
            << ",\"free_mib\":" << dev.free_mib
            << ",\"total_mib\":" << dev.total_mib
            << ",\"util_gpu\":" << dev.gpu_util
            << ",\"util_mem\":" << dev.mem_util << ",\"temp_c\":" << dev.temp_c
            << ",\"power_mw\":" << dev.power_mw
            << ",\"clk_gfx\":" << dev.clock_gfx << ",\"clk_sm\":" << dev.clock_sm
            << ",\"clk_mem\":" << dev.clock_mem
            << ",\"throttle_pwr\":" << (dev.throttle_power ? 1 : 0)
            << ",\"throttle_therm\":" << (dev.throttle_thermal ? 1 : 0)
            << ",\"pcie_rx_bw\":" << dev.pcie_rx_bps
            << ",\"pcie_tx_bw\":" << dev.pcie_tx_bps << "}";
    }
    oss << "]";
    return oss.str();
}

// --- LogChannel Implementation ---

Logger::LogChannel::LogChannel(std::string name, Options opt)
    : name_(std::move(name)), opt_(std::move(opt)) {
    if (!opt_.base_path.empty()) {
        opened_ = true;
        ensureOpenLocked();
    }
}

Logger::LogChannel::~LogChannel() { close(); }

// This is the function causing your linker error:
void Logger::LogChannel::close() {
    std::lock_guard<std::mutex> lk(mu_);
    closeLocked();
}

void Logger::LogChannel::closeLocked() {
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    opened_ = false;
}

bool Logger::LogChannel::isOpen() const { return opened_; }

std::string Logger::LogChannel::makePathLocked() const {
    std::ostringstream oss;
    // Naming format: base_path.category.index.log
    oss << "." << name_ << "." << index_ << ".log";
    return opt_.base_path + oss.str();
}

void Logger::LogChannel::ensureOpenLocked() {
    if (!opened_) return;
    if (stream_.is_open()) return;

    const std::string path = makePathLocked();
    const fs::path p(path);

    // Ensure the parent directory exists (just in case)
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
    }

    stream_.open(p, std::ios::out | std::ios::app);

    if (!stream_.good()) {
        GFL_LOG_ERROR("Failed to open log file: ", p.string());
    } else {
        current_bytes_ = 0;
    }
}

void Logger::LogChannel::rotateLocked() {
    if (!opened_) return;
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    index_ += 1;
    current_bytes_ = 0;
    ensureOpenLocked();
}

void Logger::LogChannel::write(const std::string& line) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!opened_) {
        GFL_LOG_ERROR("Write failed: Channel '", name_, "' is not opened");
        return;
    }

    ensureOpenLocked();
    if (!stream_.good()) {
        GFL_LOG_ERROR("[GPUFL-DEBUG] Write failed: Stream bad for '", name_,
                      "' (Check Permissions/Path)");
        return;
    }

    const size_t bytesToWrite = line.size() + 1;  // + '\n'
    if (opt_.rotate_bytes > 0 &&
        (current_bytes_ + bytesToWrite) > opt_.rotate_bytes) {
        rotateLocked();
        if (!stream_.good()) {
            GFL_LOG_ERROR("[GPUFL-DEBUG] Write failed: Stream bad for '", name_,
                          "' (Check Permissions/Path)");
            return;
        }
    }
    stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream_.put('\n');
    current_bytes_ += bytesToWrite;

    if (opt_.flush_always) {
        stream_.flush();
    }
}

Logger::Logger() = default;
Logger::~Logger() { close(); }

bool Logger::open(const Options& opt) {
    close();
    opt_ = opt;

    if (opt_.base_path.empty()) return false;

    // Create channels for specific categories
    chanDevice_ = std::make_unique<LogChannel>("device", opt_);
    chanScope_ = std::make_unique<LogChannel>("scope", opt_);
    chanSystem_ = std::make_unique<LogChannel>("system", opt_);

    return true;
}

void Logger::close() {
    if (chanDevice_) chanDevice_->close();
    if (chanScope_) chanScope_->close();
    if (chanSystem_) chanSystem_->close();

    chanDevice_.reset();
    chanScope_.reset();
    chanSystem_.reset();
}

// --- Broadcast Lifecycle Events ---

std::string Logger::hostToJson(const HostSample& h) {
    std::ostringstream oss;
    oss.precision(1);  // One decimal place for CPU is enough
    oss << std::fixed << "{" << "\"cpu_pct\":" << h.cpu_util_percent
        << ",\"ram_used_mib\":" << h.ram_used_mib
        << ",\"ram_total_mib\":" << h.ram_total_mib << "}";
    return oss.str();
}

void Logger::logInit(const InitEvent& e) const {
    std::ostringstream oss;
    oss << "{" << "\"type\":\"init\"" << ",\"pid\":" << e.pid << ",\"app\":\""
        << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"log_path\":\""
        << jsonEscape(e.log_path) << "\"" << ",\"ts_ns\":" << e.ts_ns
        << ",\"system_rate_ms\":" << opt_.system_sample_rate_ms
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices)
        << ",\"cuda_static_devices\":"
        << cudaStaticDeviceToJson(e.cuda_static_device_infos) << "}";

    std::string json = oss.str();

    // Broadcast to all active channels
    if (chanDevice_) chanDevice_->write(json);
    if (chanScope_) chanScope_->write(json);
    if (chanSystem_) chanSystem_->write(json);
}

void Logger::logShutdown(const ShutdownEvent& e) const {
    std::ostringstream oss;
    oss << "{" << "\"type\":\"shutdown\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"ts_ns\":" << e.ts_ns << "}";

    std::string json = oss.str();

    // Broadcast to all active channels
    if (chanDevice_) chanDevice_->write(json);
    if (chanScope_) chanScope_->write(json);
    if (chanSystem_) chanSystem_->write(json);
}

// --- Specific Event Channels ---

void Logger::logKernelEvent(const KernelEvent& e) const {
    if (!chanDevice_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"kernel_event\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"platform\":\""
        << jsonEscape(e.platform) << "\""
        << ",\"has_details\":" << std::boolalpha << e.has_details
        << ",\"device_id\":\"" << e.device_id << "\"" << ",\"stream_id\":\""
        << e.stream_id << "\"" << ",\"stack_trace\":\""
        << jsonEscape(e.stack_trace) << "\"" << ",\"user_scope\":\""
        << jsonEscape(e.user_scope) << "\""
        << ",\"scope_depth\":" << e.scope_depth << ",\"start_ns\":" << e.start_ns
        << ",\"end_ns\":" << e.end_ns << ",\"api_start_ns\":" << e.api_start_ns
        << ",\"api_exit_ns\":" << e.api_exit_ns << ",\"grid\":\""
        << jsonEscape(e.grid) << "\"" << ",\"block\":\"" << jsonEscape(e.block)
        << "\"" << ",\"dyn_shared_bytes\":" << e.dyn_shared_bytes
        << ",\"num_regs\":" << e.num_regs
        << ",\"static_shared_bytes\":" << e.static_shared_bytes
        << ",\"local_bytes\":" << e.local_bytes
        << ",\"const_bytes\":" << e.const_bytes
        << ",\"occupancy\":" << e.occupancy
        << ",\"reg_occupancy\":" << e.reg_occupancy
        << ",\"smem_occupancy\":" << e.smem_occupancy
        << ",\"warp_occupancy\":" << e.warp_occupancy
        << ",\"block_occupancy\":" << e.block_occupancy
        << ",\"limiting_resource\":\"" << jsonEscape(e.limiting_resource) << "\""
        << ",\"max_active_blocks\":" << e.max_active_blocks
        << ",\"corr_id\":" << e.corr_id
        << ",\"local_mem_total\":" << e.local_mem_total
        << ",\"cache_config_requested\":"
        << static_cast<int>(e.cache_config_requested)
        << ",\"cache_config_executed\":"
        << static_cast<int>(e.cache_config_executed)
        << ",\"shared_mem_executed\":" << e.shared_mem_executed << "}";
    chanDevice_->write(oss.str());
}

void Logger::logMemcpyEvent(const MemcpyEvent& e) const {
    if (!chanDevice_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"memcpy_event\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"platform\":\""
        << jsonEscape(e.platform) << "\"" << ",\"device_id\":\"" << e.device_id
        << "\"" << ",\"stream_id\":\"" << e.stream_id << "\""
        << ",\"stack_trace\":\"" << jsonEscape(e.stack_trace) << "\""
        << ",\"user_scope\":\"" << jsonEscape(e.user_scope) << "\""
        << ",\"scope_depth\":" << e.scope_depth << ",\"start_ns\":" << e.start_ns
        << ",\"end_ns\":" << e.end_ns << ",\"api_start_ns\":" << e.api_start_ns
        << ",\"api_exit_ns\":" << e.api_exit_ns << ",\"corr_id\":" << e.corr_id
        << ",\"bytes\":" << e.bytes << ",\"copy_kind\":\""
        << jsonEscape(e.copy_kind) << "\"" << ",\"src_kind\":\""
        << jsonEscape(e.src_kind) << "\"" << ",\"dst_kind\":\""
        << jsonEscape(e.dst_kind) << "\"" << "}";
    chanDevice_->write(oss.str());
}

void Logger::logMemsetEvent(const MemsetEvent& e) const {
    if (!chanDevice_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"memset_event\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"platform\":\""
        << jsonEscape(e.platform) << "\"" << ",\"device_id\":\"" << e.device_id
        << "\"" << ",\"stream_id\":\"" << e.stream_id << "\""
        << ",\"stack_trace\":\"" << jsonEscape(e.stack_trace) << "\""
        << ",\"user_scope\":\"" << jsonEscape(e.user_scope) << "\""
        << ",\"scope_depth\":" << e.scope_depth << ",\"start_ns\":" << e.start_ns
        << ",\"end_ns\":" << e.end_ns << ",\"api_start_ns\":" << e.api_start_ns
        << ",\"api_exit_ns\":" << e.api_exit_ns << ",\"corr_id\":" << e.corr_id
        << ",\"bytes\":" << e.bytes << "}";
    chanDevice_->write(oss.str());
}

void Logger::logScopeBegin(const ScopeBeginEvent& e) const {
    if (!chanScope_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"scope_begin\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"tag\":\"" << jsonEscape(e.tag)
        << "\"" << ",\"ts_ns\":" << e.ts_ns << ",\"user_scope\":\""
        << jsonEscape(e.user_scope) << "\""
        << ",\"scope_depth\":" << e.scope_depth
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices) << "}";
    chanScope_->write(oss.str());
}

void Logger::logScopeEnd(const ScopeEndEvent& e) const {
    if (!chanScope_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"scope_end\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"tag\":\"" << jsonEscape(e.tag)
        << "\"" << ",\"ts_ns\":" << e.ts_ns << ",\"user_scope\":\""
        << jsonEscape(e.user_scope) << "\""
        << ",\"scope_depth\":" << e.scope_depth
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices) << "}";
    chanScope_->write(oss.str());
}

void Logger::logProfileSample(const ProfileSampleEvent& e) const {
    if (!chanScope_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"profile_sample\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"ts_ns\":" << e.ts_ns
        << ",\"device_id\":" << e.device_id << ",\"corr_id\":" << e.corr_id;

    if (e.samples_count > 0) {
        oss << ",\"sample_count\":" << e.samples_count
            << ",\"stall_reason\":" << e.stall_reason << ",\"reason_name\":\""
            << jsonEscape(e.reason_name) << "\"";
    }

    if (!e.metric_name.empty()) {
        oss << ",\"metric_name\":\"" << jsonEscape(e.metric_name) << "\""
            << ",\"metric_value\":" << e.metric_value << ",\"pc_offset\":\"0x"
            << std::hex << e.pc_offset << std::dec << "\"";
    }

    oss << ",\"source_file\":\"" << jsonEscape(e.source_file) << "\""
        << ",\"function_name\":\"" << jsonEscape(e.function_name) << "\""
        << ",\"source_line\":" << e.source_line << "}";
    chanScope_->write(oss.str());
}

void Logger::logSystemSample(const SystemSampleEvent& e) const {
    if (!chanSystem_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"system_sample\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"ts_ns\":" << e.ts_ns
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices) << "}";
    chanSystem_->write(oss.str());
}

void Logger::logSystemStart(const SystemStartEvent& e) const {
    if (!chanSystem_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"system_start\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"ts_ns\":" << e.ts_ns
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices) << "}";
    chanSystem_->write(oss.str());
}

void Logger::logSystemStop(const SystemStopEvent& e) const {
    if (!chanSystem_) return;
    std::ostringstream oss;
    oss << "{" << "\"type\":\"system_stop\"" << ",\"pid\":" << e.pid
        << ",\"app\":\"" << jsonEscape(e.app) << "\"" << ",\"session_id\":\""
        << jsonEscape(e.session_id) << "\"" << ",\"name\":\""
        << jsonEscape(e.name) << "\"" << ",\"ts_ns\":" << e.ts_ns
        << ",\"host\":" << hostToJson(e.host)
        << ",\"devices\":" << devicesToJson(e.devices) << "}";
    chanSystem_->write(oss.str());
}
}  // namespace gpufl
