#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>

namespace gpufl {

/**
 * Durable, agent-uploadable control records for lifecycle state.
 *
 * These records are intentionally separate from the normal NDJSON transport
 * windows: an agent can send them through the backend's small Postgres-only
 * control endpoint while telemetry windows wait for the data-plane queue.
 * The journal is opt-in at Logger construction time; callers must not enable
 * it until the launching agent has advertised the control-plane protocol.
 */
class LifecycleControlJournal {
   public:
    // This versions the outer control-delivery envelope, not the lifecycle
    // event JSON stored byte-for-byte in payload_json (which has its own
    // event version).
    static constexpr std::uint32_t kEnvelopeSchemaVersion = 1;
    static constexpr std::size_t kMaxPayloadBytes = 64 * 1024;

    LifecycleControlJournal(std::filesystem::path session_dir,
                            std::string session_id);

    LifecycleControlJournal(const LifecycleControlJournal&) = delete;
    LifecycleControlJournal& operator=(const LifecycleControlJournal&) = delete;

    /**
     * Append one immutable lifecycle envelope. The payload is the exact JSON
     * line that was already written to the normal session log. A failed append
     * never changes normal telemetry logging; the ordinary data-plane event
     * remains the compatibility fallback.
     */
    bool append(std::string_view event_type, std::string_view payload_json);

    static std::filesystem::path controlPath(
        const std::filesystem::path& session_dir, std::size_t sequence);
    static std::filesystem::path acknowledgementPath(
        const std::filesystem::path& session_dir, std::size_t sequence);

   private:
    std::size_t nextSequenceLocked_();
    static bool isLifecycleType_(std::string_view event_type);

    std::filesystem::path session_dir_;
    std::string session_id_;
    std::size_t next_sequence_ = 0;
    std::mutex mutex_;
};

}  // namespace gpufl
