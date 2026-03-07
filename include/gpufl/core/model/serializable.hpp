#pragma once

#include <string>

namespace gpufl {

enum class Channel { Device, Scope, System, All };

struct IJsonSerializable {
    virtual std::string buildJson() const = 0;
    virtual Channel channel() const = 0;
    virtual ~IJsonSerializable() = default;
};

}  // namespace gpufl
