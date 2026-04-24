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
    // The default sink is always the FileLogSink — preserving the
    // durable on-disk NDJSON contract every existing consumer depends
    // on (agent daemon, Python analyzer, text_report). Additional
    // sinks are attached by init() depending on user options (e.g.
    // HttpLogSink when opts.remote_upload is true).
    addSink(std::make_unique<FileLogSink>(opt_));
    return true;
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
