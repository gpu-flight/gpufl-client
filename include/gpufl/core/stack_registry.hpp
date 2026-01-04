#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <functional>

namespace gpufl {
    class StackRegistry {
    public:
        static StackRegistry& instance() {
            static StackRegistry inst;
            return inst;
        }

        size_t getOrRegister(const std::string& stackTrace) {
            const size_t hash = std::hash<std::string>{}(stackTrace);

            // Optimization: Check without lock first (if using a read/write lock)
            // For simplicity here, we use a standard mutex.
            std::lock_guard<std::mutex> lock(mu_);
            if (storage_.find(hash) == storage_.end()) {
                storage_[hash] = stackTrace;
                newEntries_.push_back(hash); // Track what needs to be flushed to logs
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
}