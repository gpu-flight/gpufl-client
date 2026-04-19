#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for NvtxMarkerEvent. Emitted to the Scope channel so
 * NVTX markers naturally align with scope_event_batch events on the
 * backend — NVTX markers are semantically scope-like (named regions
 * with start/end timestamps).
 */
struct NvtxMarkerModel final : IJsonSerializable {
    explicit NvtxMarkerModel(const NvtxMarkerEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const NvtxMarkerEvent& e_;
};

}  // namespace gpufl::model
