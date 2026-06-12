#include "gpufl/core/logger/logger.hpp"

#include "gpufl/core/logger/file_log_sink.hpp"
#include "gpufl/core/logger/log_sink.hpp"

namespace gpufl {

Logger::Logger() = default;
Logger::~Logger() { close(); }

bool Logger::open(const Options& opt) {
    close();
    opt_ = opt;
    if (opt_.base_path.empty()) return false;
    // The default sink is always the FileLogSink - preserving the
    // durable on-disk NDJSON contract every consumer depends on
    // (gpufl::uploadLogs deferred upload, gpufl-agent, Python analyzer,
    // text_report). The historical HttpLogSink (live streaming) was
    // removed; sinks are now strictly local. addSink stays here for
    // test recorders and future format adapters.
    auto sink = std::make_unique<FileLogSink>(opt_);
    // Check that at least one channel file actually opened before
    // declaring success. If create_directories or stream open failed
    // (typically EACCES on a Docker volume mount left over from a
    // previous container build), the sink would otherwise silently
    // drop every write and downstream init steps (Monitor::Initialize,
    // CUPTI start, sampler thread) would proceed against a logger
    // that can't persist anything - eventually deref'ing broken state
    // and killing the Python kernel. Returning false here lets
    // gpufl::init() surface the failure cleanly as a False return to
    // the caller, with the underlying fs error already on stderr.
    const bool opened = sink->anyChannelOpen();
    addSink(std::move(sink));
    return opened;
}

void Logger::close() {
    std::lock_guard<std::mutex> lk(sinks_mu_);
    for (auto& sink : sinks_) {
        if (sink) sink->close();
    }
    sinks_.clear();
}

void Logger::addSink(std::unique_ptr<ILogSink> sink) {
    if (!sink) return;
    std::lock_guard<std::mutex> lk(sinks_mu_);
    sinks_.push_back(std::move(sink));
}

void Logger::write(const IJsonSerializable& model) {
    const std::string json = model.buildJson();
    const Channel ch = model.channel();
    std::lock_guard<std::mutex> lk(sinks_mu_);
    for (auto& sink : sinks_) {
        if (sink) sink->write(ch, json);
    }
}

}  // namespace gpufl
