#pragma once

#include <string>

namespace gpufl {

// Sass = bulky artifacts (SASS disassembly listings, source file content)
// in their own sass.log so they can't bloat the device event stream past
// upload caps. All = device+scope+system lifecycle fan-out (NOT sass).
enum class Channel { Device, Scope, System, Sass, All };

struct IJsonSerializable {
    virtual std::string buildJson() const = 0;
    virtual Channel channel() const = 0;
    virtual ~IJsonSerializable() = default;
};

}  // namespace gpufl
