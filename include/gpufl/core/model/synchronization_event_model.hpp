#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for SynchronizationEvent.
 *
 * Emitted to the Scope channel for the same reasons NvtxMarkerModel
 * uses it: synchronizations are semantically scope-like (named region
 * with start/end), arrive at mid volume (hundreds-to-thousands per
 * session), and a downstream backend ingestor that's already consuming
 * the Scope NDJSON file gets sync events for free without a new
 * channel subscription.
 *
 * The wire shape is intentionally close to NvtxMarkerModel's so the
 * Java backend's BatchIngestionServiceImpl can dispatch on `"type"`
 * cheaply — see "synchronization_event" handler.
 */
struct SynchronizationEventModel final : IJsonSerializable {
    explicit SynchronizationEventModel(const SynchronizationEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const SynchronizationEvent& e_;
};

}  // namespace gpufl::model
