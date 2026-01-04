#include "gpufl/core/scope_registry.hpp"

namespace gpufl {
    std::vector<std::string>& getThreadScopeStack() {
        thread_local std::vector<std::string> stack;
        return stack;
    }
}