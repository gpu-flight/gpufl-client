#include "gpufl/core/debug_logger.hpp"

namespace gpufl {
static std::atomic<bool> g_debugEnabled{false};

void DebugLogger::setEnabled(bool enabled) { g_debugEnabled.store(enabled); }

bool DebugLogger::isEnabled() { return g_debugEnabled.load(); }
}  // namespace gpufl
