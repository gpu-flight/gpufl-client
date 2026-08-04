#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

/** Serializes one static CUDA graph node to the low-volume Scope channel. */
struct GraphNodeDefinitionModel final : IJsonSerializable {
    explicit GraphNodeDefinitionModel(const GraphNodeDefinitionEvent& event)
        : event_(event) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

private:
    const GraphNodeDefinitionEvent& event_;
};

}  // namespace gpufl::model
