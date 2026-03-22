#pragma once

#include <string>

#include "gpufl/core/backend_interfaces.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl {

BackendCollectors CreateBackendCollectors(BackendKind backend,
                                          std::string* reasonOut);

}  // namespace gpufl
