#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpufl/core/stack_trace.hpp"

namespace gpufl {
class StackRegistry {
   public:
    static StackRegistry& instance() {
        // Leaky singleton: heap-allocated and intentionally never destroyed.
        //
        // A function-local `static StackRegistry inst;` registers its destructor
        // via atexit at FIRST USE. On Windows, first use happens during the run
        // - AFTER the injection library registered its own shutdown atexit at
        // init. atexit runs LIFO, so the registry would be destroyed BEFORE
        // gpufl::shutdown()'s teardown drain runs, and get()/getOrRegister()
        // would then touch a destroyed unordered_map (access violation -
        // observed as the Windows trace teardown crash that dropped all kernel
        // rows). Leaking the object keeps it valid for the whole process
        // lifetime; the OS reclaims the memory at exit regardless.
        static StackRegistry* inst = new StackRegistry();
        return *inst;
    }

    // Hot path (CUPTI callback / launch thread): register a captured RawStack
    // (return addresses only). NO symbolization happens here - resolution is
    // deferred to get(), which runs on the collector/worker thread. This keeps
    // dbghelp's SymFromAddr (a process-global lock) off the per-launch path.
    size_t getOrRegister(const core::RawStack& raw) {
        const size_t hash = hashRaw(raw);
        if (hash == 0) return 0;

        std::lock_guard<std::mutex> lock(mu_);
        if (storage_.find(hash) == storage_.end()) {
            Entry e;
            e.raw = raw;
            storage_.emplace(hash, std::move(e));
            newEntries_.push_back(hash);
        }
        return hash;
    }

    // Legacy: register an already-symbolized string (stored as pre-resolved).
    // Kept so callers that already hold a resolved trace don't break.
    size_t getOrRegister(const std::string& stackTrace) {
        const size_t hash = std::hash<std::string>{}(stackTrace);

        std::lock_guard<std::mutex> lock(mu_);
        if (storage_.find(hash) == storage_.end()) {
            Entry e;
            e.resolved = stackTrace;
            e.isResolved = true;
            storage_.emplace(hash, std::move(e));
            newEntries_.push_back(hash);
        }
        return hash;
    }

    // Cold path (collector/worker thread): resolve the RawStack to a sanitized
    // string, cached after first resolution. Symbol resolution runs OUTSIDE the
    // lock so it never blocks hot-path getOrRegister() calls.
    std::string get(const size_t hash) {
        if (hash == 0) return "";

        core::RawStack raw;
        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto it = storage_.find(hash);
            if (it == storage_.end()) return "unknown";
            if (it->second.isResolved) return it->second.resolved;
            raw = it->second.raw;
        }

        // Heavy symbolization happens here, without the lock held.
        std::string resolved = core::ResolveCallStack(raw);

        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto it = storage_.find(hash);
            if (it != storage_.end() && !it->second.isResolved) {
                it->second.resolved = resolved;
                it->second.isResolved = true;
            }
        }
        return resolved;
    }

    std::vector<size_t> consumeNewEntries() {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<size_t> result;
        result.swap(newEntries_);
        return result;
    }

   private:
    struct Entry {
        core::RawStack raw;     // pending frames (used when !isResolved)
        std::string resolved;   // cached symbolized+sanitized string
        bool isResolved = false;
    };

    // FNV-1a over the raw return addresses. Identical call sites produce
    // identical address sequences → same id (dedup preserved). Distinct
    // sequences that symbolize to the same string still dedup downstream at
    // internFunction(), so no wire-level duplication results.
    static size_t hashRaw(const core::RawStack& raw) {
        if (raw.count == 0) return 0;
        size_t h = 1469598103934665603ULL;
        for (uint8_t i = 0; i < raw.count; ++i) {
            h ^= reinterpret_cast<size_t>(raw.frames[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    std::mutex mu_;
    std::unordered_map<size_t, Entry> storage_;
    std::vector<size_t> newEntries_;
};
}  // namespace gpufl
