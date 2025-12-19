#include "gpufl/core/logger.hpp"
#include <sstream>
#include <memory>
#include <iostream>
#include <filesystem>

namespace gpufl {
    namespace fs = std::filesystem;

    static inline std::string jsonEscape(const std::string& s) {
        std::ostringstream oss;
        for (char c : s) {
            switch (c) {
                case '\\': oss << "\\\\"; break;
                case '"':  oss << "\\\""; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
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

    static std::string devicesToJson(const std::vector<DeviceSample>& devs) {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (auto &dev : devs) {
            if (!first) oss << ",";
            first = false;
            oss << "{"
                << "\"id\":" << dev.deviceId
                << ",\"name\":\"" << jsonEscape(dev.name) << "\""
                << ",\"uuid\":\"" << jsonEscape(dev.uuid) << "\""
                << ",\"pci_bus\":" << dev.pciBusId
                << ",\"used_mib\":" << dev.usedMiB
                << ",\"free_mib\":" << dev.freeMiB
                << ",\"total_mib\":" << dev.totalMiB
                << ",\"util_gpu\":" << dev.gpuUtil
                << ",\"util_mem\":" << dev.memUtil
                << ",\"temp_c\":" << dev.tempC
                << ",\"power_mw\":" << dev.powerMw
                << ",\"clk_gfx\":" << dev.clockGfx
                << ",\"clk_sm\":" << dev.clockSm
                << ",\"clk_mem\":" << dev.clockMem
                << "}";
        }
        oss << "]";
        return oss.str();
    }

    // --- LogChannel Implementation ---

    Logger::LogChannel::LogChannel(std::string name, Options opt)
        : name_(std::move(name)), opt_(std::move(opt)) {
        if (!opt_.basePath.empty()) {
            opened_ = true;
            ensureOpenLocked();
        }
    }

    Logger::LogChannel::~LogChannel() {
        close();
    }

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

    bool Logger::LogChannel::isOpen() const {
        return opened_;
    }

    std::string Logger::LogChannel::makePathLocked() const {
        std::ostringstream oss;
        // Naming format: basePath.category.index.log
        oss << "." << name_ << "." << index_ << ".log";
        return opt_.basePath + oss.str();
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
            std::cerr << "[GPUFL] ERROR: Failed to open log file: " << p.string() << std::endl;
        } else {
            currentBytes_ = 0;
        }
    }

    void Logger::LogChannel::rotateLocked() {
        if (!opened_) return;
        if (stream_.is_open()) {
            stream_.flush();
            stream_.close();
        }
        index_ += 1;
        currentBytes_ = 0;
        ensureOpenLocked();
    }

    void Logger::LogChannel::write(const std::string& line) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!opened_) return;

        ensureOpenLocked();
        if (!stream_.good()) return;

        const size_t bytesToWrite = line.size() + 1; // + '\n'
        if (opt_.rotateBytes > 0 && (currentBytes_ + bytesToWrite) > opt_.rotateBytes) {
            rotateLocked();
            if (!stream_.good()) return;
        }

        stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
        stream_.put('\n');
        currentBytes_ += bytesToWrite;

        if (opt_.flushAlways) {
            stream_.flush();
        }
    }

    // --- Logger Implementation ---

    Logger::Logger() = default;
    Logger::~Logger() { close(); }

    bool Logger::open(const Options& opt) {
        close();
        opt_ = opt;

        if (opt_.basePath.empty()) return false;

        // Create channels for specific categories
        chanKernel_ = std::make_unique<LogChannel>("kernel", opt_);
        chanScope_  = std::make_unique<LogChannel>("scope", opt_);
        chanSystem_ = std::make_unique<LogChannel>("system", opt_);

        return true;
    }

    void Logger::close() {
        if (chanKernel_) chanKernel_->close();
        if (chanScope_) chanScope_->close();
        if (chanSystem_) chanSystem_->close();

        chanKernel_.reset();
        chanScope_.reset();
        chanSystem_.reset();
    }

    // --- Broadcast Lifecycle Events ---


    std::string Logger::hostToJson(const HostSample& h) {
        std::ostringstream oss;
        oss.precision(1); // One decimal place for CPU is enough
        oss << std::fixed
            << "{"
            << "\"cpu_pct\":" << h.cpuUtilPercent
            << ",\"ram_used_mib\":" << h.ramUsedMiB
            << ",\"ram_total_mib\":" << h.ramTotalMiB
            << "}";
        return oss.str();
    }

    void Logger::logInit(const InitEvent& e) const {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"init\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"logPath\":\"" << jsonEscape(e.logPath) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"system_rate_ms\":" << opt_.systemSampleRateMs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";

        std::string json = oss.str();

        // Broadcast to all active channels
        if (chanKernel_) chanKernel_->write(json);
        if (chanScope_)  chanScope_->write(json);
        if (chanSystem_) chanSystem_->write(json);
    }

    void Logger::logShutdown(const ShutdownEvent& e) const {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"shutdown\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << "}";

        std::string json = oss.str();

        // Broadcast to all active channels
        if (chanKernel_) chanKernel_->write(json);
        if (chanScope_)  chanScope_->write(json);
        if (chanSystem_) chanSystem_->write(json);
    }

    // --- Specific Event Channels ---

    void Logger::logKernelBegin(const KernelBeginEvent& e) const {
        if (!chanKernel_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"kernel\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"devices\":" << devicesToJson(e.devices)
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_start_ns\":" << e.tsStartNs
            << ",\"ts_end_ns\":" << e.tsEndNs
            << ",\"duration_ns\":" << e.durationNs
            << ",\"grid\":\"" << jsonEscape(e.grid) << "\""
            << ",\"block\":\"" << jsonEscape(e.block) << "\""
            << ",\"dyn_shared_bytes\":" << e.dynSharedBytes
            << ",\"num_regs\":" << e.numRegs
            << ",\"static_shared_bytes\":" << e.staticSharedBytes
            << ",\"local_bytes\":" << e.localBytes
            << ",\"const_bytes\":" << e.constBytes
            << ",\"cuda_error\":\"" << jsonEscape(e.cudaError) << "\""
            << "}";
        chanKernel_->write(oss.str());
    }

    void Logger::logKernelEnd(const KernelEndEvent& e) const {
        if (!chanKernel_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"kernel_end\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"cuda_error\":\"" << jsonEscape(e.cudaError) << "\""
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanKernel_->write(oss.str());
    }

    void Logger::logKernelSample(const KernelSampleEvent& e) const {
        if (!chanKernel_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"kernel_sample\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanKernel_->write(oss.str());
    }

    void Logger::logScopeBegin(const ScopeBeginEvent& e) const {
        if (!chanScope_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_begin\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanScope_->write(oss.str());
    }

    void Logger::logScopeEnd(const ScopeEndEvent& e) const {
        if (!chanScope_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_end\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanScope_->write(oss.str());
    }

    void Logger::logScopeSample(const ScopeSampleEvent& e) const {
        if (!chanScope_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_sample\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanScope_->write(oss.str());
    }

    void Logger::logSystemSample(const SystemSampleEvent& e) const {
        if (!chanSystem_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_sample\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanSystem_->write(oss.str());
    }

    void Logger::logSystemStart(const SystemStartEvent &e) const {
        if (!chanSystem_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_start\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanSystem_->write(oss.str());
    }

    void Logger::logSystemStop(const SystemStopEvent &e) const {
        if (!chanSystem_) return;
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_stop\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"host\":" << hostToJson(e.host)
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        chanSystem_->write(oss.str());
    }
}
