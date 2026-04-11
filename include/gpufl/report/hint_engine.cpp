#include "gpufl/report/hint_engine.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace gpufl {
namespace report {

std::vector<std::string> computeHints(const FuncProfile& fp) {
    std::vector<std::string> hints;

    // Memory efficiency hint — three-tier fallback:
    // Tier 1: aggregate ideal (smsp__sass_sectors_mem_global_ideal, sm_120+)
    // Tier 2: sum of per-op ideals (op_ld + op_st, sm_86 fallback)
    // Tier 3: Long Scoreboard proxy when no ideal metric is available at all
    const uint64_t effectiveIdeal = fp.idealSectors > 0
        ? fp.idealSectors
        : fp.idealLoadSectors + fp.idealStoreSectors;

    if (fp.globalSectors > 0 && effectiveIdeal > 0) {
        const double memEff =
            static_cast<double>(effectiveIdeal) / fp.globalSectors * 100;
        if (memEff < 50)
            hints.push_back("Low memory efficiency (" +
                            std::to_string(static_cast<int>(memEff)) +
                            "%) - consider coalesced access patterns or shared "
                            "memory tiling.");
    } else if (fp.globalSectors > 0 && effectiveIdeal == 0) {
        auto it = fp.stalls.find("Long Scoreboard");
        if (it != fp.stalls.end() && fp.totalStalls > 0) {
            const double lsPct = it->second * 100.0 / fp.totalStalls;
            if (lsPct > 20)
                hints.push_back("High Long Scoreboard stalls (" +
                                std::to_string(static_cast<int>(lsPct)) +
                                "%) suggest memory latency; check global access "
                                "coalescing (memory efficiency metric unavailable "
                                "on this GPU).");
        }
    }

    // Stall-based hints
    if (fp.totalStalls > 0) {
        uint64_t memStalls = 0, pipeStalls = 0, execStalls = 0;
        for (const auto& [r, c] : fp.stalls) {
            std::string lower = r;
            for (auto& ch : lower)
                ch = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(ch)));
            if (lower.find("memory") != std::string::npos ||
                lower.find("mem") != std::string::npos)
                memStalls += c;
            if (lower.find("pipe") != std::string::npos) pipeStalls += c;
            if (lower.find("execution") != std::string::npos ||
                lower.find("exec") != std::string::npos)
                execStalls += c;
        }
        const double memPct = memStalls * 100.0 / fp.totalStalls;
        const double pipePct = pipeStalls * 100.0 / fp.totalStalls;
        const double execPct = execStalls * 100.0 / fp.totalStalls;

        if (memPct > 30)
            hints.push_back("Memory stalls dominate (" +
                            std::to_string(static_cast<int>(memPct)) +
                            "%) - optimize memory access patterns.");
        if (pipePct > 20)
            hints.push_back("Pipe busy stalls (" +
                            std::to_string(static_cast<int>(pipePct)) +
                            "%) - arithmetic unit contention; consider "
                            "reducing instruction intensity.");
        if (execPct > 30)
            hints.push_back("Execution dependency stalls (" +
                            std::to_string(static_cast<int>(execPct)) +
                            "%) - increase instruction-level parallelism.");
    }

    // Warp efficiency hint
    if (fp.warpInsts > 0 && fp.threadInsts > 0) {
        const double eff =
            static_cast<double>(fp.threadInsts) / fp.warpInsts / 32.0 * 100;
        if (eff < 90)
            hints.push_back("Low warp efficiency (" +
                            std::to_string(static_cast<int>(eff)) +
                            "%) - reduce branch divergence within warps.");
    }

    return hints;
}

}  // namespace report
}  // namespace gpufl
