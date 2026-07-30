#include "gpufl/core/logger/file_log_sink.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <sstream>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/log_rotator.hpp"
#include "gpufl/core/logger/log_salvage.hpp"
#include "gpufl/core/logger/session_ownership.hpp"

namespace gpufl {
namespace fs = std::filesystem;

// --- FileChannel --- (behavior preserved from pre-refactor Logger::LogChannel)

FileLogSink::FileChannel::FileChannel(std::string name, Logger::Options opt,
                                      FileLogSink* owner)
    : name_(std::move(name)), opt_(std::move(opt)), owner_(owner) {
    if (opt_.compress_rotated) {
        compressor_ = std::make_unique<GzipFileCompressor>();
    }
    LogRotationOptions r{};
    r.base_path = opt_.base_path;
    r.session_id = opt_.session_id;
    r.channel_name = name_;
    r.compress_rotated = opt_.compress_rotated;
    rotator_ = std::make_unique<LogFileRotator>(r, compressor_.get());

    // v1.2+ requires a session_id. Without it the rotator would write
    // to "<base>/<empty>/channel.log" - a path with an empty path
    // component, which `fs::create_directories` would happily make as
    // just "<base>/", landing files at parent level. The uploader
    // would then flag the parent-level files as the legacy flat
    // layout and refuse to upload. Fail loudly here instead.
    if (!opt_.base_path.empty() && !opt_.session_id.empty()) {
        opened_ = true;
        ensureOpenLocked();
        // ensureOpenLocked logs the real fs error and bails on failure
        // (e.g. EACCES from a Docker volume mount whose existing
        // contents belong to a different UID). Reflect that failure in
        // opened_ so subsequent write() calls short-circuit and so
        // FileLogSink::anyChannelOpen() can report the truth up to
        // Logger::open() → gpufl::init(). Without this flip, init()
        // returned true with a sink that silently dropped every event.
        if (!stream_.is_open()) {
            opened_ = false;
        }
    } else if (!opt_.base_path.empty()) {
        GFL_LOG_ERROR("FileLogSink: session_id is required (got empty). "
                      "Channel '", name_, "' will not be opened.");
    }
}

FileLogSink::FileChannel::~FileChannel() { close(); }

void FileLogSink::FileChannel::close() {
    std::lock_guard lk(mu_);
    closeLocked();
}

void FileLogSink::FileChannel::closeLocked() {
    const WindowTiming timing{
        window_first_write_ms_,
        window_first_write_ms_ >= 0 ? nowMs() : -1};
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    // v1.2+: compress the active .log → .log.gz on clean shutdown so
    // the finished session directory contains only compressed files.
    // The rotator's compressActive() is a no-op when the active file
    // doesn't exist (channel never wrote a byte) or compression is
    // disabled, so it's safe to call unconditionally here.
    //
    // On crash (process killed without a clean Logger::close()), this
    // path doesn't run - the uploader's lazy crash-repair branch
    // gzips the orphan .log on first read instead. That keeps the
    // on-wire format uniform regardless of whether shutdown ran.
    if (rotator_) {
        if (next_window_index_ == 0) {
            next_window_index_ = rotator_->nextWindowIndex();
        }
        (void)rotator_->compressActive(next_window_index_, timing);
    }
    opened_ = false;
}

bool FileLogSink::FileChannel::isOpen() const { return opened_; }

std::uint64_t FileLogSink::FileChannel::currentBytes() const {
    std::lock_guard lk(mu_);
    return current_bytes_;
}

void FileLogSink::FileChannel::ensureOpenLocked() {
    if (!opened_) return;
    if (stream_.is_open()) return;

    const std::string path = rotator_->activePath();
    const fs::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        // Surface the actual fs error. The most common failure here
        // is EACCES on a pre-existing log_path mounted from the host
        // (or carried over from a prior container image build) whose
        // contents are owned by a UID the current container can't
        // write to. Without this message, init() either crashed
        // downstream or silently dropped every event with no clue
        // why. Return early so we don't immediately try to open() a
        // file in a directory we know doesn't exist.
        if (ec) {
            GFL_LOG_ERROR(
                "Failed to create log directory '",
                p.parent_path().string(), "' for channel '", name_,
                "': ", ec.message(), " (error code ", ec.value(), "). ",
                "Common causes: directory exists with restrictive "
                "permissions (e.g. a Docker volume mount from a "
                "previous container image - chown the path to the "
                "current container's UID, or delete the directory "
                "and let gpufl recreate it), read-only filesystem, "
                "out of inodes.");
            return;
        }
    }
    stream_.open(p, std::ios::out | std::ios::app);
    if (!stream_.good()) {
        GFL_LOG_ERROR(
            "Failed to open log file '", p.string(), "' for channel '",
            name_, "': ofstream not in a good state after open(). ",
            "Check that the parent directory exists and is writable "
            "by the current process UID.");
    } else {
        std::error_code ec;
        current_bytes_ = fs::exists(p, ec) ? fs::file_size(p, ec) : 0;
    }
}

std::int64_t FileLogSink::FileChannel::nowMs() const {
    if (opt_.now_ms) return opt_.now_ms();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void FileLogSink::FileChannel::rotateLocked(RotateTrigger trigger) {
    if (!opened_) return;
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
    // CUTOVER ONLY - a rename, no compression, no retry sleeps. Whoever
    // hit the boundary (the collector beat, or a writer that crossed
    // rotate_bytes) must not wait on gzip: the collector would stop
    // draining the CUPTI ring, and a writer would hold this channel's
    // lock against every other event producer.
    // Allocate from this channel's own counter. Seeded once, lazily, so the
    // seed reflects anything a salvage pass published before the first
    // rotation; after that the filesystem never decides an index again (see
    // next_window_index_ for why re-deriving it races the export worker).
    if (next_window_index_ == 0) {
        next_window_index_ = rotator_->nextWindowIndex();
    }
    const std::size_t index = next_window_index_;
    const WindowTiming timing{window_first_write_ms_, nowMs()};
    const auto result = rotator_->retireActiveWindow(index);
    const char* trigger_name =
        trigger == RotateTrigger::Size ? "size" : "time";
    switch (result) {
        case LogFileRotator::RetireResult::Retired: {
            // Consumed only on success: a Blocked cutover leaves the data in
            // the active window, so the same index must be retried.
            ++next_window_index_;
            const std::uint64_t retired_bytes =
                current_bytes_;
            if (trigger == RotateTrigger::Size) {
                ++rotation_stats_.by_size;
            } else {
                ++rotation_stats_.by_time;
            }
            window_first_write_ms_ = -1;
            GFL_LOG_DEBUG("[Logger] cut over '", name_, "' window ", index,
                          ": trigger=", trigger_name,
                          " bytes=", current_bytes_);
            ensureOpenLocked();  // fresh, empty active window
            // Enqueue AFTER reopening so the channel is immediately
            // writable again even if the worker starts exporting at once.
            if (owner_) {
                owner_->enqueueRetired(this, index, retired_bytes, timing);
            }
            return;
        }
        case LogFileRotator::RetireResult::Blocked:
            // The data is still the active window. Keep its first-write
            // age so the very next beat retries instead of waiting out a
            // fresh rotate_after_ms.
            ++rotation_stats_.cutover_blocked;
            break;
        case LogFileRotator::RetireResult::NoData:
            break;
    }
    ensureOpenLocked();
}

void FileLogSink::FileChannel::exportRetired(
    const std::size_t index, const WindowTiming timing) {
    if (opt_.before_retired_export) opt_.before_retired_export();
    const auto started = std::chrono::steady_clock::now();
    // No channel lock held: the retired file is immutable and nothing
    // writes to it any more, so gzip and the publish backoff cost this
    // worker thread only.
    const auto result = rotator_->exportRetiredWindow(index, timing);
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started)
            .count();

    std::lock_guard lk(mu_);
    if (elapsed_ms > rotation_stats_.max_export_ms) {
        rotation_stats_.max_export_ms = elapsed_ms;
    }
    switch (result) {
        case LogFileRotator::ExportWindowResult::Published:
            ++rotation_stats_.published;
            break;
        case LogFileRotator::ExportWindowResult::StagedForSalvage:
            // Compressed but not published: the salvage pass finishes it.
            ++rotation_stats_.staged;
            break;
        case LogFileRotator::ExportWindowResult::DeferredInActive:
            // Compression failed; the retired file stays in `.tmp` and the
            // salvage pass compresses it at close.
            ++rotation_stats_.export_failed;
            break;
        case LogFileRotator::ExportWindowResult::NoData:
            break;
    }
}

bool FileLogSink::FileChannel::timeDueLocked(const std::int64_t now) const {
    return opt_.rotate_after_ms > 0 && window_first_write_ms_ >= 0 &&
           now - window_first_write_ms_ >= opt_.rotate_after_ms;
}

void FileLogSink::FileChannel::rotateIfDue() {
    std::lock_guard lk(mu_);
    if (!opened_) return;
    if (!timeDueLocked(nowMs())) return;
    rotateLocked(RotateTrigger::Time);
}

bool FileLogSink::FileChannel::write(std::string_view line) {
    std::lock_guard lk(mu_);
    if (!opened_) {
        GFL_LOG_ERROR("Write failed: Channel '", name_, "' is not opened");
        return false;
    }
    ensureOpenLocked();
    if (!stream_.good()) {
        GFL_LOG_ERROR("Write failed: Stream bad for '", name_, "'");
        return false;
    }
    const size_t bytesToWrite = line.size() + 1;
    // Two rotation triggers, whichever is due first. Time is evaluated
    // before size: an overdue window was already due BEFORE this write's
    // bytes existed, so it is the honest "which came first" answer when
    // both hold at once. The time trigger only ever fires on a NON-empty
    // window (window_first_write_ms_ >= 0), so idle channels never
    // publish empty window files, and the monotonic clock (nowMs) means
    // wall-time jumps can neither fire nor starve it. This write-path
    // check gives immediacy on busy channels; quiet channels are covered
    // by the collector beat calling rotateIfDue().
    // Time-triggered channels need `now` on every write to test the
    // deadline. Other channels avoid that hot-path clock read after their
    // first write; they still capture one timestamp per non-empty window so
    // the immutable transport metadata is complete even when rotation is
    // size-only or disabled.
    const std::int64_t now =
        opt_.rotate_after_ms > 0 ? nowMs() : -1;
    const bool time_due = timeDueLocked(now);
    const bool size_due =
        opt_.rotate_bytes > 0 &&
        (current_bytes_ + bytesToWrite) > opt_.rotate_bytes;
    if (time_due || size_due) {
        rotateLocked(time_due ? RotateTrigger::Time : RotateTrigger::Size);
        if (!stream_.good()) {
            GFL_LOG_ERROR("Write failed after rotate for '", name_, "'");
            return false;
        }
    }
    // Write line + newline. Per-write flush is gated behind
    // Options::flush_always (wired from InitOptions::flush_logs_always)
    // because it's a real syscall in the hot path and matters under
    // high event volume - SASS-heavy PyTorch sessions can push tens of
    // thousands of batched lines per second across channels, at which
    // point the fsync cost dominates.
    //
    // When OFF (default): rely on libc's ofstream buffer (~4 KiB) +
    // OS page cache. Rotation (rotateLocked, above) and shutdown
    // (closeLocked) explicitly flush, so the .log → .log.gz pipeline
    // and the uploader's lazy crash-repair path both still see complete
    // data. Worst case on SIGKILL: a few buffered NDJSON lines lost -
    // acceptable for profiling data.
    //
    // When ON: flush at every line boundary so a live tailer (the agent's
    // LogTailer in the Phase 4 live-push design) never observes a
    // partial mid-record line. Without this, the tailer's readLine() can
    // hit a buffered-but-unflushed write, get truncated JSON, and Jackson
    // throws FAIL_ON_TRAILING_TOKENS errors.
    stream_.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream_.put('\n');
    if (opt_.flush_always) {
        stream_.flush();
    }
    if (window_first_write_ms_ < 0) {
        // A size-triggered cutover may have reset the timestamp after `now`
        // was conditionally sampled above. In the non-time-triggered case,
        // read the monotonic clock here so the new window records its own
        // first write rather than 0 or its predecessor's timestamp.
        window_first_write_ms_ = now >= 0 ? now : nowMs();
    }
    current_bytes_ += bytesToWrite;
    return true;
}

FileLogSink::RotationStats FileLogSink::FileChannel::rotationStats() const {
    std::lock_guard lk(mu_);
    return rotation_stats_;
}

// --- FileLogSink ---

FileLogSink::FileLogSink(const Logger::Options& opt) {
    if (opt.base_path.empty()) return;
    if (opt.session_id.empty()) {
        GFL_LOG_ERROR("FileLogSink: session_id is required for session "
                      "ownership.");
        return;
    }
    const fs::path session_dir = fs::path(opt.base_path) / opt.session_id;
    session_dir_ = session_dir;
    max_spool_bytes_ = opt.max_spool_bytes;
    min_free_bytes_ = opt.min_free_bytes;
    spool_now_ms_ = opt.now_ms;
    std::string lock_error;
    session_ownership_ =
        SessionOwnershipLock::tryAcquire(session_dir, &lock_error);
    if (!session_ownership_) {
        GFL_LOG_ERROR("FileLogSink: cannot own session directory '",
                      session_dir.string(), "': ", lock_error,
                      ". Refusing to open channels so two live processes "
                      "cannot mutate the same spool.");
        return;
    }
    chanDevice_ = std::make_unique<FileChannel>("device", opt, this);
    chanScope_  = std::make_unique<FileChannel>("scope",  opt, this);
    chanSystem_ = std::make_unique<FileChannel>("system", opt, this);
    chanSass_   = std::make_unique<FileChannel>("sass",   opt, this);
    LogRotationOptions r{};
    r.base_path = opt.base_path;
    r.session_id = opt.session_id;
    temp_dir_ = LogFileRotator(r, nullptr).tempDir();
    spool_estimated_bytes_.store(
        spoolBytesOnDisk(), std::memory_order_relaxed);
    // Establish the real filesystem reserve before the first profiling event.
    // Waiting for the collector beat lets a large first event fill an already
    // constrained filesystem before the guard ever runs.
    checkSpoolBudget(/*force=*/true);
}

FileLogSink::~FileLogSink() { close(); }

bool FileLogSink::anyChannelOpen() const {
    return (chanDevice_ && chanDevice_->isOpen()) ||
           (chanScope_  && chanScope_->isOpen())  ||
           (chanSystem_ && chanSystem_->isOpen()) ||
           (chanSass_   && chanSass_->isOpen());
}

FileLogSink::RotationStats FileLogSink::rotationStats() const {
    RotationStats total;
    for (const FileChannel* ch :
         {chanDevice_.get(), chanScope_.get(), chanSystem_.get(),
          chanSass_.get()}) {
        if (!ch) continue;
        const RotationStats s = ch->rotationStats();
        total.by_size += s.by_size;
        total.by_time += s.by_time;
        total.cutover_blocked += s.cutover_blocked;
        total.published += s.published;
        total.staged += s.staged;
        total.export_failed += s.export_failed;
        total.max_export_ms =
            std::max(total.max_export_ms, s.max_export_ms);
    }
    {
        std::lock_guard lk(retire_mu_);
        total.pending_exports = pending_exports_;
        total.max_pending_exports = max_pending_exports_;
        total.pending_export_bytes = pending_export_bytes_;
        total.max_pending_export_bytes = max_pending_export_bytes_;
        total.lost_windows = lost_windows_;
    }
    total.spool_saturated =
        spool_saturated_.load(std::memory_order_acquire);
    total.spool_bytes_at_saturation =
        spool_bytes_at_saturation_.load(std::memory_order_relaxed);
    total.dropped_events =
        dropped_events_.load(std::memory_order_relaxed);
    total.dropped_bytes =
        dropped_bytes_.load(std::memory_order_relaxed);
    return total;
}

std::uint64_t FileLogSink::spoolBytesOnDisk() const {
    std::uint64_t total = 0;
    std::error_code ec;
    if (session_dir_.empty() || !fs::exists(session_dir_, ec)) return 0;
    fs::recursive_directory_iterator it(
        session_dir_, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end) {
        std::error_code entry_ec;
        if (it->is_regular_file(entry_ec)) {
            const fs::path path = it->path();
            const bool active_channel =
                path.parent_path().filename() == ".tmp" &&
                (path.filename() == "device.log" ||
                 path.filename() == "scope.log" ||
                 path.filename() == "system.log" ||
                 path.filename() == "sass.log");
            if (!active_channel) {
                const auto bytes = it->file_size(entry_ec);
                if (!entry_ec) total += static_cast<std::uint64_t>(bytes);
            }
        }
        it.increment(ec);
    }
    for (const FileChannel* ch :
         {chanDevice_.get(), chanScope_.get(), chanSystem_.get(),
          chanSass_.get()}) {
        if (ch) total += ch->currentBytes();
    }
    return total;
}

void FileLogSink::markSpoolSaturated(
    const std::uint64_t spool_bytes,
    const std::uint64_t available_bytes,
    const char* reason) {
    if (spool_saturated_.exchange(true, std::memory_order_acq_rel)) return;
    spool_bytes_at_saturation_.store(
        spool_bytes, std::memory_order_relaxed);

    std::ostringstream detail;
    detail << reason << ":spool_bytes=" << spool_bytes
           << ",max_spool_bytes=" << max_spool_bytes_
           << ",available_bytes=" << available_bytes
           << ",min_free_bytes=" << min_free_bytes_;
    const bool marker_written = recordTransportLossMarker(
        session_dir_, "spool", 0, detail.str());
    GFL_LOG_ERROR(
        "[Logger] transport spool saturated for session '",
        session_dir_.string(), "' (", detail.str(),
        "). GPUFlight stopped accepting new profiling events to protect "
        "the target application's filesystem. The profile is incomplete.",
        marker_written
            ? " A durable transport-loss marker was written."
            : " WARNING: the durable loss marker could not be written.");
}

std::int64_t FileLogSink::spoolNowMs() const {
    if (spool_now_ms_) return spool_now_ms_();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void FileLogSink::checkSpoolBudget(
    const bool force, const std::uint64_t incoming_bytes) {
    if (spool_saturated_.load(std::memory_order_acquire) ||
        (max_spool_bytes_ == 0 && min_free_bytes_ == 0)) {
        return;
    }
    std::lock_guard lk(spool_budget_mu_);
    if (spool_saturated_.load(std::memory_order_relaxed)) return;
    const std::int64_t now = spoolNowMs();
    // Directory scans are intentionally not on the 250 ms collector cadence.
    // One scan per second reconciles agent-side ACK deletions and filesystem
    // free space; the cheap write-byte estimate forces an immediate scan when
    // this process approaches its session limit.
    if (!force && last_spool_check_ms_ >= 0 &&
        now - last_spool_check_ms_ < 1000) {
        return;
    }
    last_spool_check_ms_ = now;

    const std::uint64_t spool_bytes = spoolBytesOnDisk();
    spool_estimated_bytes_.store(spool_bytes, std::memory_order_relaxed);
    std::error_code space_ec;
    const auto space = fs::space(session_dir_, space_ec);
    const std::uint64_t available =
        space_ec ? std::numeric_limits<std::uint64_t>::max()
                 : static_cast<std::uint64_t>(space.available);
    available_bytes_estimate_.store(available, std::memory_order_relaxed);
    const auto reaches = [](const std::uint64_t current,
                            const std::uint64_t added,
                            const std::uint64_t limit) {
        return current >= limit || added >= limit - current;
    };
    if (max_spool_bytes_ > 0 &&
        reaches(spool_bytes, incoming_bytes, max_spool_bytes_)) {
        const std::uint64_t projected =
            incoming_bytes >
                    std::numeric_limits<std::uint64_t>::max() - spool_bytes
                ? std::numeric_limits<std::uint64_t>::max()
                : spool_bytes + incoming_bytes;
        markSpoolSaturated(
            projected, available, "spool_budget_exceeded");
    } else if (min_free_bytes_ > 0 && !space_ec &&
               (available <= min_free_bytes_ ||
                incoming_bytes >= available - min_free_bytes_)) {
        markSpoolSaturated(spool_bytes, available, "filesystem_reserve_reached");
    }
}

void FileLogSink::rotateDueWindows() {
    checkSpoolBudget();
    for (FileChannel* ch : {chanDevice_.get(), chanScope_.get(),
                            chanSystem_.get(), chanSass_.get()}) {
        if (ch) ch->rotateIfDue();
    }
}

void FileLogSink::enqueueRetired(FileChannel* channel,
                                 const std::size_t index,
                                 const std::uint64_t bytes,
                                 const WindowTiming timing) {
    if (!channel) return;
    std::lock_guard lk(retire_mu_);
    if (retire_stop_) {
        // Never export inline here: enqueueRetired() is called while the
        // channel mutex is held, and exportRetired() takes that same mutex to
        // fold stats, which would self-deadlock. The indexed raw file is
        // already durable in `.tmp`; close-time salvage owns this rare race.
        GFL_LOG_WARN("[Logger] retirement worker already stopped; leaving "
                     "window ", index, " in `.tmp` for salvage.");
        return;
    }
    retire_queue_.push_back({channel, index, bytes, timing});
    ++pending_exports_;
    pending_export_bytes_ += bytes;
    max_pending_exports_ =
        std::max(max_pending_exports_, pending_exports_);
    max_pending_export_bytes_ =
        std::max(max_pending_export_bytes_, pending_export_bytes_);
    if (pending_exports_ >= 8 &&
        (pending_exports_ & (pending_exports_ - 1)) == 0) {
        GFL_LOG_WARN("[Logger] retirement export backlog: ",
                     pending_exports_, " window(s), ",
                     pending_export_bytes_, " raw byte(s) pending.");
    }
    if (!retire_worker_.joinable()) {
        // Started on first use: a session that never rotates never pays
        // for a thread.
        retire_worker_ = std::thread([this] {
            for (;;) {
                RetiredWindow item{};
                {
                    std::unique_lock lk(retire_mu_);
                    retire_cv_.wait(lk, [this] {
                        return retire_stop_ || !retire_queue_.empty();
                    });
                    if (retire_queue_.empty()) return;  // stop + drained
                    item = retire_queue_.front();
                    retire_queue_.pop_front();
                    ++exports_in_flight_;
                }
                // Outside every lock: gzip + publish retries live here.
                item.channel->exportRetired(item.index, item.timing);
                {
                    std::lock_guard lk(retire_mu_);
                    --exports_in_flight_;
                    if (pending_exports_ > 0) --pending_exports_;
                    if (pending_export_bytes_ >= item.bytes) {
                        pending_export_bytes_ -= item.bytes;
                    } else {
                        pending_export_bytes_ = 0;
                    }
                }
                retire_cv_.notify_all();
            }
        });
    }
    retire_cv_.notify_one();
}

void FileLogSink::waitForPendingExports() {
    std::unique_lock lk(retire_mu_);
    retire_cv_.wait(lk, [this] {
        return retire_queue_.empty() && exports_in_flight_ == 0;
    });
}

void FileLogSink::stopRetirementWorker() {
    {
        std::lock_guard lk(retire_mu_);
        retire_stop_ = true;
    }
    retire_cv_.notify_all();
    if (retire_worker_.joinable()) retire_worker_.join();
}

void FileLogSink::close() {
    // Drain and stop the worker BEFORE the channels go away: it holds raw
    // channel pointers, and window indices must stop moving before each
    // channel exports its final active window.
    stopRetirementWorker();
    if (chanDevice_) chanDevice_->close();
    if (chanScope_)  chanScope_->close();
    if (chanSystem_) chanSystem_->close();
    if (chanSass_)   chanSass_->close();
    chanDevice_.reset();
    chanScope_.reset();
    chanSystem_.reset();
    chanSass_.reset();
    // Every channel has exported and dropped its active file - the
    // shared session temp dir can go now (loud on failure, swept by the
    // launcher's salvage pass otherwise).
    if (!temp_dir_.empty()) {
        const fs::path session_dir = fs::path(temp_dir_).parent_path();
        const auto salvage = salvageOwnedSessionTempDir(session_dir);
        if (salvage.lost_windows > 0) {
            // Terminal: those events are gone. Say so where an operator will
            // see it AND keep it in the stats, because everything downstream
            // (an emptied `.tmp`, a finished upload) now looks like a clean
            // session. Reporting this onward is the open follow-up.
            std::lock_guard lk(retire_mu_);
            lost_windows_ += static_cast<std::size_t>(salvage.lost_windows);
            GFL_LOG_ERROR("[Logger] session '", session_dir.string(), "': ",
                          salvage.lost_windows,
                          " durable transport-loss marker(s) are present. "
                          "Some profiling events are LOST and the uploaded "
                          "session must remain incomplete.");
        }
        if (salvage.deferred > 0 || sessionTempDirHasDeferredData(session_dir)) {
            GFL_LOG_ERROR("[Logger] session temp dir '", temp_dir_,
                          "' still contains deferred log data (salvaged=",
                          salvage.salvaged, ", deferred=", salvage.deferred,
                          ") - leaving it for the next salvage pass.");
            temp_dir_.clear();
            session_ownership_.reset();
            return;
        }
        std::error_code ec;
        fs::remove_all(temp_dir_, ec);
        if (ec) {
            GFL_LOG_ERROR("[Logger] could not remove session temp dir '",
                          temp_dir_, "' (", ec.message(),
                          ") - safe to delete once no process holds it.");
        }
        temp_dir_.clear();
    }
    // Release only after every active/raw/staged artifact has been reconciled
    // and `.tmp` has either been removed or deliberately left visible.
    session_ownership_.reset();
}

FileLogSink::FileChannel* FileLogSink::resolveChannel(Channel ch) const {
    switch (ch) {
        case Channel::Device: return chanDevice_.get();
        case Channel::Scope:  return chanScope_.get();
        case Channel::System: return chanSystem_.get();
        case Channel::Sass:   return chanSass_.get();
        default:              return nullptr;
    }
}

void FileLogSink::write(Channel ch, std::string_view json) {
    if (spool_saturated_.load(std::memory_order_acquire)) {
        dropped_events_.fetch_add(1, std::memory_order_relaxed);
        dropped_bytes_.fetch_add(
            json.size() + 1,
            std::memory_order_relaxed);
        return;
    }
    const std::uint64_t planned_copies =
        ch == Channel::All
            ? static_cast<std::uint64_t>(
                  (chanDevice_ ? 1 : 0) + (chanScope_ ? 1 : 0) +
                  (chanSystem_ ? 1 : 0) + (chanSass_ ? 1 : 0))
            : (resolveChannel(ch) ? 1u : 0u);
    if (planned_copies == 0) return;
    const std::uint64_t event_bytes = json.size() + 1;
    const std::uint64_t incoming_bytes =
        event_bytes >
                std::numeric_limits<std::uint64_t>::max() / planned_copies
            ? std::numeric_limits<std::uint64_t>::max()
            : event_bytes * planned_copies;
    const std::uint64_t estimated =
        spool_estimated_bytes_.load(std::memory_order_relaxed);
    const std::uint64_t available_estimate =
        available_bytes_estimate_.load(std::memory_order_relaxed);
    const bool near_spool_limit =
        max_spool_bytes_ > 0 &&
        (estimated >= max_spool_bytes_ ||
         incoming_bytes >= max_spool_bytes_ - estimated);
    const bool needs_space_baseline =
        min_free_bytes_ > 0 &&
        available_estimate == std::numeric_limits<std::uint64_t>::max();
    const bool near_space_reserve =
        min_free_bytes_ > 0 && !needs_space_baseline &&
        (available_estimate <= min_free_bytes_ ||
         incoming_bytes >= available_estimate - min_free_bytes_);
    if (near_spool_limit || needs_space_baseline || near_space_reserve) {
        checkSpoolBudget(/*force=*/true, incoming_bytes);
        if (spool_saturated_.load(std::memory_order_acquire)) {
            dropped_events_.fetch_add(1, std::memory_order_relaxed);
            dropped_bytes_.fetch_add(event_bytes, std::memory_order_relaxed);
            return;
        }
    }
    std::uint64_t copies_written = 0;
    if (ch == Channel::All) {
        if (chanDevice_ && chanDevice_->write(json)) ++copies_written;
        if (chanScope_ && chanScope_->write(json)) ++copies_written;
        if (chanSystem_ && chanSystem_->write(json)) ++copies_written;
        // Sass is part of the fan-out so each sass.log is self-ordered:
        // dictionary_update lines land in the file BEFORE the
        // source_file_content/cubin_disassembly lines that reference their
        // IDs - live tailers (agent) read channels independently and have
        // no cross-file ordering.
        if (chanSass_ && chanSass_->write(json)) ++copies_written;
    } else {
        if (FileChannel* channel = resolveChannel(ch)) {
            if (channel->write(json)) ++copies_written;
        }
    }
    if (copies_written > 0) {
        const std::uint64_t bytes =
            copies_written * (json.size() + 1);
        spool_estimated_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        auto available =
            available_bytes_estimate_.load(std::memory_order_relaxed);
        while (available != std::numeric_limits<std::uint64_t>::max()) {
            const std::uint64_t reduced =
                bytes >= available ? 0 : available - bytes;
            if (available_bytes_estimate_.compare_exchange_weak(
                    available, reduced, std::memory_order_relaxed)) {
                break;
            }
        }
    }
}

}  // namespace gpufl
