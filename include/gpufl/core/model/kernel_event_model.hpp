#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct KernelEventModel final : IJsonSerializable {
    explicit KernelEventModel(const KernelEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }
private:
    const KernelEvent& e_;
};

}  // namespace gpufl::model
