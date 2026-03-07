#pragma once

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

struct ProfileSampleModel final : IJsonSerializable {
    explicit ProfileSampleModel(const ProfileSampleEvent& e) : e_(e) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }
private:
    const ProfileSampleEvent& e_;
};

}  // namespace gpufl::model
