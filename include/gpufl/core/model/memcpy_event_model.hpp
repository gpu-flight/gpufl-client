#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct MemcpyEventModel final : IJsonSerializable {
    explicit MemcpyEventModel(const MemcpyEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }
private:
    const MemcpyEvent& e_;
};

struct MemsetEventModel final : IJsonSerializable {
    explicit MemsetEventModel(const MemsetEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }
private:
    const MemsetEvent& e_;
};

}  // namespace gpufl::model
