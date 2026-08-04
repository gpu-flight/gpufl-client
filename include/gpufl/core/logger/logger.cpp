#include "gpufl/core/logger/logger.hpp"

#include "gpufl/core/logger/file_log_sink.hpp"
#include "gpufl/core/logger/lifecycle_control_journal.hpp"
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
    if (opened && opt_.lifecycle_control_journal_enabled) {
        lifecycle_control_journal_ = std::make_unique<LifecycleControlJournal>(
            std::filesystem::path(opt_.base_path) / opt_.session_id,
            opt_.session_id);
    }
    return opened;
}

void Logger::setSerializedBytesCallbackBeforeFirstWrite(
    std::function<void(std::uint64_t)> callback) {
    std::lock_guard lock(sinks_mu_);
    opt_.on_serialized_bytes = callback;
    for (auto& sink : sinks_) {
        if (sink) {
            sink->setSerializedBytesCallbackBeforeFirstWrite(callback);
        }
    }
}

void Logger::close() {
    std::lock_guard<std::mutex> lk(sinks_mu_);
    for (auto& sink : sinks_) {
        if (sink) sink->close();
    }
    sinks_.clear();
    lifecycle_control_journal_.reset();
}

void Logger::addSink(std::unique_ptr<ILogSink> sink) {
    if (!sink) return;
    std::lock_guard<std::mutex> lk(sinks_mu_);
    sinks_.push_back(std::move(sink));
}

void Logger::rotateDueWindows() {
    std::lock_guard<std::mutex> lk(sinks_mu_);
    for (auto& sink : sinks_) {
        if (sink) sink->rotateDueWindows();
    }
}

void Logger::write(const IJsonSerializable& model) {
    const std::string json = model.buildJson();
    const Channel ch = model.channel();
    std::lock_guard<std::mutex> lk(sinks_mu_);
    for (auto& sink : sinks_) {
        if (sink) sink->write(ch, json);
    }
    if (lifecycle_control_journal_) {
        const std::string_view event_type = model.lifecycleControlEventType();
        if (!event_type.empty()) {
            // The data-plane line is written first. A journal failure is
            // deliberately non-fatal: the ordinary lifecycle line remains
            // the compatible eventual-consistency path for older agents.
            (void)lifecycle_control_journal_->append(event_type, json);
        }
    }
}

}  // namespace gpufl
