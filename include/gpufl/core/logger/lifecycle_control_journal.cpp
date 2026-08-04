#include "gpufl/core/logger/lifecycle_control_journal.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <system_error>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/logger/log_salvage.hpp"

namespace gpufl {
namespace fs = std::filesystem;
namespace {

constexpr std::string_view kControlPrefix = ".gpufl-control.";
constexpr std::string_view kAcknowledgementPrefix = ".gpufl-control-ack.";
constexpr std::string_view kSuffix = ".json";
// A collision is only possible after an external repair or crash residue. Keep
// this bounded so a damaged journal cannot block lifecycle emission forever.
constexpr int kMaxSequenceCollisionRetries = 16;

bool updateSequenceFromName(const std::string& name,
                            const std::string_view prefix,
                            std::size_t& max_sequence) {
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size() + kSuffix.size() ||
        name.compare(name.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
        return false;
    }
    try {
        const auto sequence = static_cast<std::size_t>(std::stoull(
            name.substr(prefix.size(), name.size() - prefix.size() - kSuffix.size())));
        if (sequence == 0) return false;
        max_sequence = std::max(max_sequence, sequence);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

LifecycleControlJournal::LifecycleControlJournal(fs::path session_dir,
                                                 std::string session_id)
    : session_dir_(std::move(session_dir)), session_id_(std::move(session_id)) {}

fs::path LifecycleControlJournal::controlPath(const fs::path& session_dir,
                                               const std::size_t sequence) {
    return session_dir /
           (std::string(kControlPrefix) + std::to_string(sequence) +
            std::string(kSuffix));
}

fs::path LifecycleControlJournal::acknowledgementPath(
    const fs::path& session_dir, const std::size_t sequence) {
    return session_dir /
           (std::string(kAcknowledgementPrefix) + std::to_string(sequence) +
            std::string(kSuffix));
}

bool LifecycleControlJournal::isLifecycleType_(const std::string_view event_type) {
    return event_type == "job_start" || event_type == "segment_start" ||
           event_type == "segment_end" || event_type == "run_end" ||
           event_type == "shutdown";
}

std::size_t LifecycleControlJournal::nextSequenceLocked_() {
    if (next_sequence_ != 0) return next_sequence_;

    std::size_t max_sequence = 0;
    std::error_code ec;
    if (fs::is_directory(session_dir_, ec)) {
        for (const auto& entry : fs::directory_iterator(session_dir_, ec)) {
            std::error_code file_ec;
            if (!entry.is_regular_file(file_ec)) continue;
            const auto name = entry.path().filename().string();
            (void)updateSequenceFromName(name, kControlPrefix, max_sequence);
            (void)updateSequenceFromName(name, kAcknowledgementPrefix,
                                         max_sequence);
        }
    }
    next_sequence_ = max_sequence + 1;
    return next_sequence_;
}

bool LifecycleControlJournal::append(const std::string_view event_type,
                                     const std::string_view payload_json) {
    if (!isLifecycleType_(event_type) || session_id_.empty() ||
        payload_json.empty() || payload_json.size() > kMaxPayloadBytes) {
        GFL_LOG_ERROR("[LifecycleControl] refusing invalid control record '",
                      std::string(event_type), "' for session '", session_id_,
                      "'.");
        return false;
    }

    // The backend validates this independently, but rejecting a malformed or
    // mismatched local payload here prevents an agent from retrying a control
    // file that can never receive an acknowledgement. This runs only for the
    // five rare lifecycle records, never on the telemetry hot path.
    const auto payload = json::parseJson(std::string(payload_json));
    if (!payload.is_object() ||
        payload.value<std::string>("type", "") != event_type ||
        payload.value<std::string>("session_id", "") != session_id_) {
        GFL_LOG_ERROR("[LifecycleControl] refusing malformed or mismatched control payload for session '",
                      session_id_, "'.");
        return false;
    }

    std::lock_guard lock(mutex_);
    std::error_code dir_ec;
    fs::create_directories(session_dir_, dir_ec);
    if (dir_ec) {
        GFL_LOG_ERROR("[LifecycleControl] cannot create journal directory '",
                      session_dir_.string(), "': ", dir_ec.message());
        return false;
    }

    // A sequence collision must never overwrite an older control record. It
    // is unexpected while the SessionOwnershipLock is held, but retries with
    // the next tombstone-aware sequence keep the journal convergent after a
    // crash or external repair.
    for (int attempt = 0; attempt < kMaxSequenceCollisionRetries; ++attempt) {
        const std::size_t sequence = nextSequenceLocked_();
        const std::string control_id = detail::GenerateSessionId();
        const fs::path target = controlPath(session_dir_, sequence);
        const fs::path partial =
            target.string() + ".part." + control_id;

        {
            std::ofstream out(partial, std::ios::binary | std::ios::trunc);
            if (!out) {
                GFL_LOG_ERROR("[LifecycleControl] cannot stage control record '",
                              partial.string(), "'.");
                return false;
            }
            out << "{\"schema_version\":" << kEnvelopeSchemaVersion
                << ",\"control_id\":\""
                << json::escape(control_id) << "\",\"session_id\":\""
                << json::escape(session_id_) << "\",\"control_sequence\":"
                << sequence << ",\"event_type\":\""
                << json::escape(std::string(event_type))
                << "\",\"payload_json\":\""
                << json::escape(std::string(payload_json)) << "\"}\n";
            out.flush();
            if (!out.good()) {
                out.close();
                std::error_code remove_ec;
                fs::remove(partial, remove_ec);
                GFL_LOG_ERROR("[LifecycleControl] cannot finish control record '",
                              partial.string(), "'.");
                return false;
            }
        }

        std::error_code move_ec;
        const auto moved = moveFileNoReplace(partial, target, move_ec);
        if (moved == MoveFileNoReplaceResult::Moved) {
            ++next_sequence_;
            return true;
        }

        std::error_code remove_ec;
        fs::remove(partial, remove_ec);
        if (moved == MoveFileNoReplaceResult::DestinationExists) {
            ++next_sequence_;
            continue;
        }
        GFL_LOG_ERROR("[LifecycleControl] cannot publish control record '",
                      target.string(), "': ", move_ec.message());
        return false;
    }

    GFL_LOG_ERROR("[LifecycleControl] exhausted sequence collision retries for session '",
                  session_id_, "'.");
    return false;
}

}  // namespace gpufl
