#pragma once

#include <string>

#include "gpufl/gpufl.hpp"

namespace gpufl {

/// Loads a JSON config file and merges values into InitOptions.
/// File values fill in options; env vars and explicit code values can override later.
class ConfigFileLoader {
   public:
    /// The settings a file decided, for callers that report where a setting
    /// came from. Only settings reported that way are tracked.
    struct Applied {
        bool stack_trace = false;
    };

    /// Read the JSON file at `path` and apply matching fields to `opts`.
    /// Logs a warning and returns silently on missing/invalid files.
    static Applied apply(InitOptions& opts, const std::string& path);
};

}  // namespace gpufl
