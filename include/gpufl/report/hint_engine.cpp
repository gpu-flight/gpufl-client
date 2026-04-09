#include "gpufl/report/hint_engine.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace gpufl {
namespace report {

std::vector<std::string> computeHints(const FuncProfile& fp) {
    std::vector<std::string> hints;

    // Memory efficiency hint
    if (fp.globalSectors > 0 && fp.idealSectors > 0) {
        const double memEff =
            static_cast<double>(fp.idealSectors) / fp.globalSectors * 100;
        if (memEff < 50)
            hints.push_back("Low memory efficiency (" +
                            std::to_string(static_cast<int>(memEff)) +
                            "%) - consider coalesced access patterns or shared "
                            "memory tiling.");
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
