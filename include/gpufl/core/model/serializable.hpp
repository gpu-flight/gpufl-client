#pragma once

#include <string>
#include <string_view>

namespace gpufl {

// Sass = bulky artifacts (SASS disassembly listings, source file content)
// in their own sass.log so they can't bloat the device event stream past
// upload caps. All = device+scope+system lifecycle fan-out (NOT sass).
enum class Channel { Device, Scope, System, Sass, All };

struct IJsonSerializable {
    virtual std::string buildJson() const = 0;
    virtual Channel channel() const = 0;
    // Empty means normal telemetry. The five lifecycle models override this
    // so Logger can copy their already-serialized payload into the optional
    // priority control journal without parsing or serializing a second time.
    virtual std::string_view lifecycleControlEventType() const noexcept {
        return {};
    }
    virtual ~IJsonSerializable() = default;
};

}  // namespace gpufl
