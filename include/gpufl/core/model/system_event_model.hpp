#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct SystemStartModel final : IJsonSerializable {
    explicit SystemStartModel(const SystemStartEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::System; }
private:
    const SystemStartEvent& e_;
};

struct SystemStopModel final : IJsonSerializable {
    explicit SystemStopModel(const SystemStopEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::System; }
private:
    const SystemStopEvent& e_;
};

struct SystemSampleModel final : IJsonSerializable {
    explicit SystemSampleModel(const SystemSampleEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::System; }
private:
    const SystemSampleEvent& e_;
};

}  // namespace gpufl::model
