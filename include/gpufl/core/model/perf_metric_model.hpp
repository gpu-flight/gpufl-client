#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct PerfMetricModel final : IJsonSerializable {
    explicit PerfMetricModel(const PerfMetricEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const PerfMetricEvent& e_;
};

struct KernelPerfMetricModel final : IJsonSerializable {
    explicit KernelPerfMetricModel(const KernelPerfMetricEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }
private:
    const KernelPerfMetricEvent& e_;
};

}  // namespace gpufl::model
