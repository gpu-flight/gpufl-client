#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for MemoryAllocEvent.
 *
 * Per-event JSON, Scope channel — same rationale as the sync event
 * model: mid-volume, semantically scope-like (a discrete event on the
 * timeline), and reusing an existing channel keeps the backend
 * ingestor's subscription set small.
 *
 * Wire shape mirrors SynchronizationEventModel intentionally so the
 * Java backend can dispatch on `"type"` cheaply — see
 * "memory_alloc_event" handler.
 */
struct MemoryAllocEventModel final : IJsonSerializable {
    explicit MemoryAllocEventModel(const MemoryAllocEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const MemoryAllocEvent& e_;
};

}  // namespace gpufl::model
