#pragma once
#include <string>
#include <vector>
#include <mutex>

namespace gpufl {
    std::vector<std::string>& getScopeStack();
    std::mutex& getScopeMutex();
}
