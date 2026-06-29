#include "gpufl/backends/nvidia/cupti_activity_state.hpp"

#include <cupti.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl {
namespace {

// Persistent maps for ActivityAPI PC sampling companion records.
// SOURCE_LOCATOR records map sourceLocatorId -> (fileName, lineNumber).
// FUNCTION records map functionId -> functionName.
// Both arrive in the same buffer as PC_SAMPLING records and must outlive
// individual BufferCompleted calls.
std::mutex g_sourceLocatorMu;
std::unordered_map<uint32_t, std::pair<std::string, uint32_t>> g_sourceLocatorMap;
std::unordered_map<uint32_t, std::string> g_functionNameMap;

// NVTX marker pairing. CUPTI delivers each NVTX range as two separate
// activity records: one with flags=START, one with flags=END, both
// sharing the same id. We pair them here in the buffer-completion
// callback to emit a single NvtxMarkerEvent with start, end, and
// duration. Map entry value: (name, start_timestamp, domain).
std::mutex g_nvtxMu;
std::unordered_map<uint32_t, NvtxOpenRange> g_nvtxOpen;

// External-correlation map.
//
// CUPTI emits CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records whenever a
// framework brackets a code region with cuptiActivityPushExternalCorrelationId
// and a kernel launches inside that bracket. The record carries:
//   - externalKind  : which framework (PyTorch / TF / JAX / OPENACC / ...)
//   - externalId    : the framework's per-op id
//   - correlationId : the CUPTI per-launch id, identical to the matching
//                     KERNEL activity record's correlationId
//
// In practice the EXTERNAL_CORRELATION record arrives *before* the matching
// KERNEL record within a single buffer (CUPTI emits the bracket events as
// the launch is enqueued, before the launch completes on the GPU). When
// the kernel record arrives we look up its corr_id in this map, stamp the
// (kind, id) onto the kernel's ActivityRecord, and erase from the map so
// it doesn't leak across sessions.
//
// If a kernel arrives BEFORE its external correlation record (rare; would
// require CUPTI to deliver records out of generation order), we miss the
// stamp for that one launch - emit external_id == 0, treated by the
// dashboard as "no framework attribution." Acceptable best-effort.
struct ExternalCorrInfo {
    uint8_t kind = 0;
    uint64_t id = 0;
};
std::mutex g_extCorrMu;
std::unordered_map<uint32_t, ExternalCorrInfo> g_extCorrMap;

}  // namespace

void ResetCuptiActivityCompanionState() {
    {
        std::lock_guard lk(g_sourceLocatorMu);
        g_sourceLocatorMap.clear();
        g_functionNameMap.clear();
    }
    {
        std::lock_guard lk(g_nvtxMu);
        g_nvtxOpen.clear();
    }
    ClearExternalCorrelationState();
}

void ClearExternalCorrelationState() {
    std::lock_guard lk(g_extCorrMu);
    g_extCorrMap.clear();
}

void StoreExternalCorrelation(const uint32_t corr_id, const uint8_t kind,
                              const uint64_t id) {
    std::lock_guard lk(g_extCorrMu);
    g_extCorrMap[corr_id] = ExternalCorrInfo{kind, id};
}

// Public helper for cross-TU access. KernelLaunchHandler calls this from
// `handleActivityRecord` to stamp the kernel with its framework op id. Returns
// false (and leaves outputs untouched) when no matching external correlation
// has been seen yet for this corr_id.
//
// Pop-on-read: each correlation record matches exactly one kernel, and keeping
// stale entries would slowly grow the map across long sessions.
bool LookupAndPopExternalCorrelation(const uint32_t corr_id,
                                     uint8_t* kind_out,
                                     uint64_t* id_out) {
    std::lock_guard lk(g_extCorrMu);
    const auto it = g_extCorrMap.find(corr_id);
    if (it == g_extCorrMap.end()) return false;
    if (kind_out) *kind_out = it->second.kind;
    if (id_out) *id_out = it->second.id;
    g_extCorrMap.erase(it);
    return true;
}

void StoreSourceLocator(const uint32_t id, const char* file_name,
                        uint32_t line_number) {
    if (!file_name) return;
    std::lock_guard lk(g_sourceLocatorMu);
    g_sourceLocatorMap[id] = {file_name, line_number};
}

void StoreFunctionName(uint32_t id, const char* name) {
    if (!name) return;
    std::lock_guard lk(g_sourceLocatorMu);
    g_functionNameMap[id] = name;
}

bool LookupSourceLocator(const uint32_t id, std::string* file_name,
                         uint32_t* line_number) {
    std::lock_guard lk(g_sourceLocatorMu);
    const auto it = g_sourceLocatorMap.find(id);
    if (it == g_sourceLocatorMap.end()) return false;
    if (file_name) *file_name = it->second.first;
    if (line_number) *line_number = it->second.second;
    return true;
}

bool LookupFunctionName(const uint32_t id, std::string* name) {
    std::lock_guard lk(g_sourceLocatorMu);
    const auto it = g_functionNameMap.find(id);
    if (it == g_functionNameMap.end()) return false;
    if (name) *name = it->second;
    return true;
}

void StoreNvtxMarkerStart(const uint32_t id, std::string name, std::string domain, const uint64_t start_ts) {
    NvtxOpenRange entry;
    entry.name = std::move(name);
    entry.domain = std::move(domain);
    entry.start_ts = start_ts;

    std::lock_guard lk(g_nvtxMu);
    g_nvtxOpen[id] = std::move(entry);
}

bool PopNvtxMarker(const uint32_t id, NvtxOpenRange* out) {
    if (!out) return false;
    std::lock_guard lk(g_nvtxMu);
    const auto it = g_nvtxOpen.find(id);
    if (it == g_nvtxOpen.end()) return false;
    *out = std::move(it->second);
    g_nvtxOpen.erase(it);
    return true;
}

// F1 active push: thin wrappers over CUPTI's correlation stack. The
// caller (e.g. `gpufl.torch.attach()`) calls these around a code region
// - every kernel launched in between gets the (kind, id) emitted as an
// EXTERNAL_CORRELATION record, which our BufferCompleted path then
// stamps onto the matching kernel's row. This is what makes F1 useful
// without requiring a framework profiler to be running.
//
// Both operations are pure CUPTI library calls; they don't need a
// CuptiBackend instance to exist (the stack is per-thread inside CUPTI
// itself). Safe to call before init / after shutdown - CUPTI returns
// CUPTI_ERROR_NOT_INITIALIZED which we silently ignore.
//
// Diagnostic: count pushes + log the first few + log any error result.
// "Pushes happen with OK return but no EXTERNAL_CORRELATION records"
// is a distinct failure mode from "pushes never happen" - these logs
// distinguish them. Also log the OS thread id; if pushes happen on a
// different thread than the kernel launches, CUPTI's per-thread stack
// won't bracket the launch.
void pushExternalCorrelation(uint32_t kind, uint64_t id) {
    const CUptiResult res = cuptiActivityPushExternalCorrelationId(
        static_cast<CUpti_ExternalCorrelationKind>(kind), id);
    static std::atomic g_push_count{0};
    const int n = g_push_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 5 || res != CUPTI_SUCCESS) {
        const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        GFL_LOG_DEBUG("[ExternalCorr] push #", n,
                      " kind=", kind, " id=", id,
                      " result=", static_cast<int>(res),
                      " tid=", static_cast<uint64_t>(tid));
    }
}

void popExternalCorrelation(uint32_t kind) {
    uint64_t lastId = 0;
    const CUptiResult res = cuptiActivityPopExternalCorrelationId(
        static_cast<CUpti_ExternalCorrelationKind>(kind), &lastId);
    static std::atomic g_pop_count{0};
    const int n = g_pop_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 5 || res != CUPTI_SUCCESS) {
        GFL_LOG_DEBUG("[ExternalCorr] pop #", n,
                      " kind=", kind,
                      " lastId=", lastId,
                      " result=", static_cast<int>(res));
    }
}

}  // namespace gpufl
