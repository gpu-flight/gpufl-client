#include "rocm_collector.hpp"

namespace gpufl::amd {

    RocmCollector::RocmCollector() = default;
    RocmCollector::~RocmCollector() = default;

    std::vector<gpufl::DeviceSample> RocmCollector::sampleAll() { return {}; }

} // namespace gpufl::amd
