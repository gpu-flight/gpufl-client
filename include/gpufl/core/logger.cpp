#include "gpufl/core/logger.hpp"
#include <sstream>

namespace gpufl {
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

    Logger::Logger() = default;
    Logger::~Logger() { close(); }

    bool Logger::open(const Options& opt) {
        std::lock_guard lk(mu_);
        closeLocked_();

        opt_ = opt;
        file_ = LogFileState{};
        file_.basePath = opt_.basePath;
        if (file_.basePath.empty()) return false;

        opened_ = true;
        ensureOpenLocked();
        return file_.stream.good();
    }

    void Logger::close() {
        std::lock_guard lk(mu_);
        closeLocked_();
    }

    void Logger::closeLocked_() {
        if (file_.stream.is_open()) {
            file_.stream.flush();
            file_.stream.close();
        }
        opened_ = false;
        file_ = LogFileState{};
    }

    void Logger::ensureOpenLocked() {
        if (!opened_) return;
        if (file_.stream.is_open()) return;

        const std::string path = makePathLocked();
        file_.stream.open(path, std::ios::out | std::ios::app);
        file_.currentBytes = 0;
    }

    std::string Logger::makePathLocked() const {
        std::ostringstream oss;
        oss << file_.basePath << "." << file_.index << ".log";
        return oss.str();
    }

    void Logger::rotateLocked() {
        if (!opened_) return;
        if (file_.stream.is_open()) {
            file_.stream.flush();
            file_.stream.close();
        }
        file_.index += 1;
        file_.currentBytes = 0;
        ensureOpenLocked();
    }

    void Logger::writeLine(const std::string& jsonLine) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!opened_) return;

        ensureOpenLocked();
        if (!file_.stream.good()) return;

        // Rotation check
        const size_t bytesToWrite = jsonLine.size() + 1; // + '\n'
        if (opt_.rotateBytes > 0 && (file_.currentBytes + bytesToWrite) > opt_.rotateBytes) {
            rotateLocked();
            if (!file_.stream.good()) return;
        }

        file_.stream.write(jsonLine.data(), static_cast<std::streamsize>(jsonLine.size()));
        file_.stream.put('\n');
        file_.currentBytes += bytesToWrite;

        if (opt_.flushAlways) {
            file_.stream.flush();
        }
    }

    void Logger::logInit(const InitEvent& e, const std::string& devicesJson) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"init\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"logPath\":\"" << jsonEscape(e.logPath) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesJson
            << "}";
        writeLine(oss.str());
    }

    void Logger::logShutdown(const ShutdownEvent& e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"shutdown\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << "}";
        writeLine(oss.str());
    }

    void Logger::logKernel(const KernelEvent& e, const std::string& devicesJson) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"kernel\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"devices\":" << devicesJson
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
        writeLine(oss.str());
    }

    void Logger::logScopeBegin(const ScopeBeginEvent& e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_begin\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }

    void Logger::logScopeEnd(const ScopeEndEvent& e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_end\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }

    void Logger::logScopeSample(const ScopeSampleEvent& e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"scope_sample\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"tag\":\"" << jsonEscape(e.tag) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }

    void Logger::logSystemSample(const SystemSampleEvent& e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_sample\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }

    void Logger::logSystemStart(const SystemStartEvent &e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_start\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }

    void Logger::logSystemStop(const SystemStopEvent &e) {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"system_stop\""
            << ",\"pid\":" << e.pid
            << ",\"app\":\"" << jsonEscape(e.app) << "\""
            << ",\"name\":\"" << jsonEscape(e.name) << "\""
            << ",\"ts_ns\":" << e.tsNs
            << ",\"devices\":" << devicesToJson(e.devices)
            << "}";
        writeLine(oss.str());
    }
}
