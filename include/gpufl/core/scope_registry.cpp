#include "gpufl/core/scope_registry.hpp"

namespace gpufl {
    std::vector<std::string>& getThreadScopeStack() {
        // This variable is unique per thread.
        // It is initialized the first time this line is hit by that thread.
        thread_local std::vector<std::string> stack;
        return stack;
    }
}