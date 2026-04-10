#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {

/**
 * @brief Compresses SASS disassembly by detecting runs of structurally
 *        identical instructions (differing only in immediate operands).
 *
 * Designed for extensibility — new comparison strategies (e.g.,
 * register-agnostic, predicate-agnostic) can be added as new methods
 * without touching the call sites.
 */
class SassCompressor {
   public:
    struct CompressedEntry {
        uint32_t pc;       ///< PC offset of the first instruction in the run
        std::string sass;  ///< representative SASS text (first instruction)
        uint32_t count;    ///< number of instructions (1 = no compression)
    };

    /**
     * Compress a sorted vector of (pc, sass) entries using run-length
     * encoding.  Consecutive instructions whose normalized form matches
     * are merged into a single CompressedEntry with count > 1.
     */
    static std::vector<CompressedEntry> compress(
        const std::vector<std::pair<uint32_t, std::string>>& entries);

    /**
     * Normalize a SASS instruction for structural comparison.
     * Replaces integer and floating-point literals with '#', preserving
     * the opcode, registers, and punctuation.
     *
     * Examples:
     *   "FFMA R0, R3, 42, R0"                     → "FFMA R0, R3, #, R0"
     *   "FFMA R0, R3, 1.009999990463..., R3"       → "FFMA R0, R3, #, R3"
     *   "LDG.E R5, desc[UR6][R4.64]"               → "LDG.E R5, desc[UR6][R4.#]"
     */
    static std::string normalizeForCompare(const std::string& sass);

    /**
     * Check if two instructions are "same shape" — same opcode and
     * register operands, differing only in immediate values.
     */
    static bool isSameShape(const std::string& a, const std::string& b);
};

}  // namespace gpufl
