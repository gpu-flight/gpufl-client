#pragma once

#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/backends/nvidia/cupti_common.hpp"

namespace gpufl {

/**
 * Captures a CUDA graph's static node topology and correlates each GRAPH_TRACE
 * activity row to the opaque CUgraphExec that was launched. The graph's
 * execution duration remains aggregate-only: CUDA does not expose trustworthy
 * replay timing for individual nodes through this capture path.
 */
class GraphStructureHandler final : public ICuptiHandler {
public:
    explicit GraphStructureHandler(CuptiBackend* backend);

    const char* getName() const override { return "GraphStructureHandler"; }
    bool shouldHandle(CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid) const override;
    void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                const void* cbdata) override;
    std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const override;
    void emitCurrentSegmentMetadata() override;

private:
    void captureDefinition_(CUgraph graph, uint64_t graph_exec_key);
    static bool extractInstantiation_(CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid,
                                      const void* function_params,
                                      CUgraph* graph,
                                      uint64_t* graph_exec_key_out);
    static uint64_t extractDestroyedExecKey_(CUpti_CallbackDomain domain,
                                             CUpti_CallbackId cbid,
                                             const void* function_params);
    void forgetDefinition_(uint64_t graph_exec_key);

    CuptiBackend* backend_;
    const std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> handled_;
    // Graph creation callbacks are rare. A mutex is deliberately confined to
    // this cold path; replay launches and activity-buffer processing remain
    // lock-free.
    std::mutex definitions_mu_;
    std::unordered_map<uint64_t, std::vector<ActivityRecord>> definitions_;
};

}  // namespace gpufl
