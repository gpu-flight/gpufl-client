#pragma once
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace gpufl {
class StackRegistry {
   public:
    static StackRegistry& instance() {
        // Leaky singleton: heap-allocated and intentionally never destroyed.
        //
        // A function-local `static StackRegistry inst;` registers its destructor
        // via atexit at FIRST USE. On Windows, first use happens during the run
        // — AFTER the injection library registered its own shutdown atexit at
        // init. atexit runs LIFO, so the registry would be destroyed BEFORE
        // gpufl::shutdown()'s teardown drain runs, and get()/getOrRegister()
        // would then touch a destroyed unordered_map (access violation —
        // observed as the Windows trace teardown crash that dropped all kernel
        // rows). Leaking the object keeps it valid for the whole process
        // lifetime; the OS reclaims the memory at exit regardless.
        static StackRegistry* inst = new StackRegistry();
        return *inst;
    }

    size_t getOrRegister(const std::string& stackTrace) {
        const size_t hash = std::hash<std::string>{}(stackTrace);

        // Optimization: Check without lock first (if using a read/write lock)
        // For simplicity here, we use a standard mutex.
        std::lock_guard<std::mutex> lock(mu_);
        if (storage_.find(hash) == storage_.end()) {
            storage_[hash] = stackTrace;
            newEntries_.push_back(
                hash);  // Track what needs to be flushed to logs
        }
        return hash;
    }

    std::string get(const size_t hash) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = storage_.find(hash);
        return (it != storage_.end()) ? it->second : "unknown";
    }

    std::vector<size_t> consumeNewEntries() {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<size_t> result;
        result.swap(newEntries_);
        return result;
    }

   private:
    std::mutex mu_;
    std::unordered_map<size_t, std::string> storage_;
    std::vector<size_t> newEntries_;
};
}  // namespace gpufl