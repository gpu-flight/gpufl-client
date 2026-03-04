#pragma once
#include <mutex>
#include <string>
#include <vector>

namespace gpufl {
std::vector<std::string>& getThreadScopeStack();
}
