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

struct SassConfigModel final : IJsonSerializable {
    explicit SassConfigModel(const SassConfigEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }
private:
    const SassConfigEvent& e_;
};

struct CaptureCapabilitiesModel final : IJsonSerializable {
    explicit CaptureCapabilitiesModel(const CaptureCapabilitiesEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::All; }
private:
    const CaptureCapabilitiesEvent& e_;
};

struct ExecutionSignatureModel final : IJsonSerializable {
    explicit ExecutionSignatureModel(const ExecutionSignatureEvent& e) : e_(e) {}
    std::string buildJson() const override;
    // Scope-attributed per-pass kernel-launch fingerprint — grouped with the
    // other per-scope profiling data (profile_sample_batch) on the Scope channel.
    Channel channel() const override { return Channel::Scope; }
private:
    const ExecutionSignatureEvent& e_;
};

}  // namespace gpufl::model
