#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for DeepWindowEvent. Emitted to the Scope channel: a
 * deep window is a named region with start/end timestamps, so it belongs
 * with scope_event_batch and nvtx_marker_event rather than in the
 * per-device event stream.
 */
struct DeepWindowModel final : IJsonSerializable {
    explicit DeepWindowModel(const DeepWindowEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const DeepWindowEvent& e_;
};

}  // namespace gpufl::model
