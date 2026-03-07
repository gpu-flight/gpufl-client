#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct InitEventModel final : IJsonSerializable {
    explicit InitEventModel(const InitEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::All; }
private:
    const InitEvent& e_;
};

struct ShutdownEventModel final : IJsonSerializable {
    explicit ShutdownEventModel(const ShutdownEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::All; }
private:
    const ShutdownEvent& e_;
};

}  // namespace gpufl::model
