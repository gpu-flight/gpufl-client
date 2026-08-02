#pragma once

#include <cstdint>
#include <string>

#include "gpufl/gpufl.hpp"

namespace gpufl::detail {

// Values accepted from the launcher/runtime environment before any Runtime is
// allocated. The launcher decides which values to export; this type records
// the validated contract the embedded runtime consumes.
struct StartupSegmentationOptions {
    uint64_t segment_every_ms = 0;
    uint64_t segment_max_rows = 0;
    uint64_t run_roll_every_ms = 0;
    uint64_t run_roll_max_bytes = 0;
    std::string run_id;

    bool enabled() const {
        return segment_every_ms > 0 || segment_max_rows > 0;
    }
};

// Applies the configuration-file fallback and resolves a canonical API path.
// Call before any component observes InitOptions.
void resolveStartupOptions(InitOptions& options);

// Reads and validates segmentation/rollover environment state without
// allocating a Runtime. On failure, `error` contains the user-facing reason.
bool readStartupSegmentationOptions(StartupSegmentationOptions& options,
                                    std::string& error);

}  // namespace gpufl::detail
