#include "rocm_collector.hpp"

namespace gpufl::amd {

    RocmCollector::RocmCollector() = default;
    RocmCollector::~RocmCollector() = default;

    std::string RocmCollector::devicesInventoryJson() { return "[]"; }
    std::vector<gpufl::DeviceSample> RocmCollector::sampleAll() { return {}; }

} // namespace gpufl::amd
