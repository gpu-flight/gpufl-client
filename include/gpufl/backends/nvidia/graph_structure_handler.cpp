#include "gpufl/backends/nvidia/graph_structure_handler.hpp"

#include <cstdint>

#include "gpufl/core/monitor.hpp"

namespace gpufl {
namespace {

template <typename GraphExecHandle>
uint64_t graphExecKey(const GraphExecHandle graph_exec) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(graph_exec));
}

}  // namespace

GraphStructureHandler::GraphStructureHandler(CuptiBackend* backend)
    : backend_(backend),
      handled_([this] {
          const auto callbacks = requiredCallbacks();
          return std::set(callbacks.begin(), callbacks.end());
      }()) {}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
GraphStructureHandler::requiredCallbacks() const {
    if (!backend_ || !backend_->GetOptions().enable_cuda_graphs_tracking) {
        return {};
    }
    return {
        // llama.cpp and most framework users call the CUDA Runtime API. The
        // driver callbacks below remain necessary for direct-driver clients.
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecDestroy_v10000},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy},
    };
}

uint64_t GraphStructureHandler::extractDestroyedExecKey_(
    const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
    const void* const function_params) {
    if (!function_params) return 0;
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecDestroy_v10000) {
        return graphExecKey(
            static_cast<const cudaGraphExecDestroy_v10000_params*>(function_params)->graphExec);
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
        cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy) {
        return graphExecKey(
            static_cast<const cuGraphExecDestroy_params*>(function_params)->hGraphExec);
    }
    return 0;
}

void GraphStructureHandler::forgetDefinition_(const uint64_t graph_exec_key) {
    if (graph_exec_key == 0) return;
    std::lock_guard lock(definitions_mu_);
    definitions_.erase(graph_exec_key);
}

bool GraphStructureHandler::shouldHandle(const CUpti_CallbackDomain domain,
                                         const CUpti_CallbackId cbid) const {
    return handled_.count({domain, cbid}) != 0;
}

bool GraphStructureHandler::extractInstantiation_(
    const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
    const void* const function_params, CUgraph* const graph,
    uint64_t* const graph_exec_key_out) {
    if (!function_params || !graph || !graph_exec_key_out) return false;
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        // CUDA Runtime and Driver graph handles are opaque pointer handles
        // backed by the same CUDA graph object. We only pass the graph into
        // cuGraphGetNodes and preserve the execution handle as an opaque
        // numeric key; no node timing or ABI-specific object layout is read.
        switch (cbid) {
            case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000: {
                const auto* p = static_cast<const cudaGraphInstantiate_v12000_params*>(function_params);
                if (!p->pGraphExec || !*p->pGraphExec) return false;
                *graph = reinterpret_cast<CUgraph>(p->graph);
                *graph_exec_key_out = graphExecKey(*p->pGraphExec);
                return true;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040: {
                const auto* p = static_cast<const cudaGraphInstantiateWithFlags_v11040_params*>(function_params);
                if (!p->pGraphExec || !*p->pGraphExec) return false;
                *graph = reinterpret_cast<CUgraph>(p->graph);
                *graph_exec_key_out = graphExecKey(*p->pGraphExec);
                return true;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000: {
                const auto* p = static_cast<const cudaGraphInstantiateWithParams_v12000_params*>(function_params);
                if (!p->pGraphExec || !*p->pGraphExec) return false;
                *graph = reinterpret_cast<CUgraph>(p->graph);
                *graph_exec_key_out = graphExecKey(*p->pGraphExec);
                return true;
            }
            case CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000: {
                const auto* p = static_cast<const cudaGraphInstantiateWithParams_ptsz_v12000_params*>(function_params);
                if (!p->pGraphExec || !*p->pGraphExec) return false;
                *graph = reinterpret_cast<CUgraph>(p->graph);
                *graph_exec_key_out = graphExecKey(*p->pGraphExec);
                return true;
            }
            default:
                return false;
        }
    }

    switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate: {
            const auto* p = static_cast<const cuGraphInstantiate_params*>(function_params);
            if (!p->phGraphExec || !*p->phGraphExec) return false;
            *graph = p->hGraph;
            *graph_exec_key_out = graphExecKey(*p->phGraphExec);
            return true;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2: {
            const auto* p = static_cast<const cuGraphInstantiate_v2_params*>(function_params);
            if (!p->phGraphExec || !*p->phGraphExec) return false;
            *graph = p->hGraph;
            *graph_exec_key_out = graphExecKey(*p->phGraphExec);
            return true;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags: {
            const auto* p = static_cast<const cuGraphInstantiateWithFlags_params*>(function_params);
            if (!p->phGraphExec || !*p->phGraphExec) return false;
            *graph = p->hGraph;
            *graph_exec_key_out = graphExecKey(*p->phGraphExec);
            return true;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams: {
            const auto* p = static_cast<const cuGraphInstantiateWithParams_params*>(function_params);
            if (!p->phGraphExec || !*p->phGraphExec) return false;
            *graph = p->hGraph;
            *graph_exec_key_out = graphExecKey(*p->phGraphExec);
            return true;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz: {
            const auto* p = static_cast<const cuGraphInstantiateWithParams_ptsz_params*>(function_params);
            if (!p->phGraphExec || !*p->phGraphExec) return false;
            *graph = p->hGraph;
            *graph_exec_key_out = graphExecKey(*p->phGraphExec);
            return true;
        }
        default:
            return false;
    }
}

void GraphStructureHandler::handle(const CUpti_CallbackDomain domain,
                                   const CUpti_CallbackId cbid,
                                   const void* const cbdata) {
    if (!backend_ || !backend_->IsActive()) return;
    const auto* info = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!info || !info->functionParams) return;

    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecDestroy_v10000 ||
         cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy)) {
        if (info->callbackSite == CUPTI_API_ENTER) {
            forgetDefinition_(extractDestroyedExecKey_(domain, cbid,
                                                       info->functionParams));
        }
        return;
    }

    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000) {
        if (info->callbackSite != CUPTI_API_ENTER) return;
        uint64_t graph_exec_key = 0;
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            graph_exec_key = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000
                ? graphExecKey(static_cast<const cudaGraphLaunch_v10000_params*>(
                    info->functionParams)->graphExec)
                : graphExecKey(static_cast<const cudaGraphLaunch_ptsz_v10000_params*>(
                    info->functionParams)->graphExec);
        } else {
            graph_exec_key = cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch
                ? graphExecKey(static_cast<const cuGraphLaunch_params*>(
                    info->functionParams)->hGraph)
                : graphExecKey(static_cast<const cuGraphLaunch_ptsz_params*>(
                    info->functionParams)->hGraphExec);
        }
        if (graph_exec_key == 0) return;
        ActivityRecord binding{};
        binding.type = TraceType::GRAPH_EXEC_LAUNCH;
        binding.corr_id = info->correlationId;
        binding.graph_exec_key = graph_exec_key;
        Monitor::PushActivityRecord(binding);
        return;
    }

    if (info->callbackSite != CUPTI_API_EXIT) return;
    CUgraph graph = nullptr;
    uint64_t graph_exec_key = 0;
    if (!extractInstantiation_(domain, cbid, info->functionParams, &graph,
                               &graph_exec_key) ||
        !graph || graph_exec_key == 0) {
        return;
    }
    captureDefinition_(graph, graph_exec_key);
}

void GraphStructureHandler::captureDefinition_(const CUgraph graph,
                                               const uint64_t graph_exec_key) {
    size_t count = 0;
    if (cuGraphGetNodes(graph, nullptr, &count) != CUDA_SUCCESS || count == 0) {
        return;
    }
    std::vector<CUgraphNode> nodes(count);
    if (cuGraphGetNodes(graph, nodes.data(), &count) != CUDA_SUCCESS) return;
    nodes.resize(count);

    std::vector<ActivityRecord> definition;
    definition.reserve(nodes.size());
    for (uint32_t index = 0; index < nodes.size(); ++index) {
        CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
        if (cuGraphNodeGetType(nodes[index], &type) != CUDA_SUCCESS) continue;
        size_t dependency_count = 0;
        if (cuGraphNodeGetDependencies(nodes[index], nullptr, nullptr,
                                       &dependency_count)
            != CUDA_SUCCESS) {
            dependency_count = 0;
        }
        ActivityRecord node{};
        node.type = TraceType::GRAPH_NODE_DEFINITION;
        node.graph_exec_key = graph_exec_key;
        node.graph_node_index = index;
        node.graph_node_type = static_cast<uint32_t>(type);
        node.graph_node_dependency_count = static_cast<uint32_t>(dependency_count);
        definition.push_back(node);
    }
    if (definition.empty()) return;
    {
        std::lock_guard lock(definitions_mu_);
        // CUDA Runtime and Driver callbacks can report the same instantiation.
        // The key is unique until its matching destroy callback, so only its
        // first successful definition is materialized for this process epoch.
        if (!definitions_.emplace(graph_exec_key, definition).second) return;
    }
    // Graph instantiation is a cold path. Publish this first definition through
    // the normal MPSC activity ring now, rather than relying on process teardown
    // to deliver a topology that was discovered while the collector was alive.
    // The retained copy is still re-emitted when a later segment begins.
    for (const auto& node : definition) Monitor::PushActivityRecord(node);
}

void GraphStructureHandler::emitCurrentSegmentMetadata() {
    std::vector<ActivityRecord> snapshot;
    {
        std::lock_guard lock(definitions_mu_);
        for (const auto& [_, definition] : definitions_) {
            snapshot.insert(snapshot.end(), definition.begin(), definition.end());
        }
    }
    for (const auto& node : snapshot) Monitor::PushActivityRecord(node);
}

}  // namespace gpufl
