#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct ScopeBeginModel final : IJsonSerializable {
    explicit ScopeBeginModel(const ScopeBeginEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const ScopeBeginEvent& e_;
};

struct ScopeEndModel final : IJsonSerializable {
    explicit ScopeEndModel(const ScopeEndEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const ScopeEndEvent& e_;
};

}  // namespace gpufl::model
