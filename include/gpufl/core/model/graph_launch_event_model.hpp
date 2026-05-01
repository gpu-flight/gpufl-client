#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/**
 * JSON serializer for GraphLaunchEvent.
 *
 * Per-event JSON, Scope channel — low volume,
 * scope-like timing. Wire shape mirrors the memory event pattern
 * intentionally so the Java backend can dispatch on `"type"` cheaply.
 */
struct GraphLaunchEventModel final : IJsonSerializable {
    explicit GraphLaunchEventModel(const GraphLaunchEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const GraphLaunchEvent& e_;
};

}  // namespace gpufl::model
