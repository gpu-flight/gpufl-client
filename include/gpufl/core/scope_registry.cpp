#include "gpufl/core/scope_registry.hpp"

namespace gpufl {
    static std::vector<std::string> g_globalStack;
    static std::mutex g_globalMutex;

    std::vector<std::string>& getScopeStack() {
        return g_globalStack;
    }

    std::mutex& getScopeMutex() {
        return g_globalMutex;
    }
}