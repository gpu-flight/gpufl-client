#pragma once

#include <cstdint>
#include <iomanip>
#include <ios>
#include <random>
#include <sstream>
#include <string>
#include "scope_registry.hpp"

namespace gpufl {
    inline thread_local std::vector<std::string> g_threadScopeStack;

    namespace detail {

        static std::string uuidToString(const char* bytes) {
            std::stringstream ss;
            ss << "GPU-";
            ss << std::hex << std::setfill('0');
            for (int i = 0; i < 16; ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) ss << "-";
                ss << std::setw(2) << (static_cast<unsigned int>(bytes[i]) & 0xFF);
            }
            return ss.str();
        }

        static std::string generateSessionId() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 15);
            std::uniform_int_distribution<> dis2(8, 11);

            std::stringstream ss;
            ss << std::hex;
            for (int i = 0; i < 8; i++) ss << dis(gen);
            ss << "-";
            for (int i = 0; i < 4; i++) ss << dis(gen);
            ss << "-4"; // UUID version 4
            for (int i = 0; i < 3; i++) ss << dis(gen);
            ss << "-";
            ss << dis2(gen); // UUID variant
            for (int i = 0; i < 3; i++) ss << dis(gen);
            ss << "-";
            for (int i = 0; i < 12; i++) ss << dis(gen);
            return ss.str();
        }
        static bool isInternalFunction(const std::string& name) {
            if (name.rfind("__device_stub", 0) == 0) return true;      // Starts with __device_stub
            if (name.find("__cudaLaunch") != std::string::npos) return true;
            if (name.find("_cudaGetProc") != std::string::npos) return true;
            if (name.find("cupti") != std::string::npos) return true;
            if (name.find("driver") != std::string::npos) return true; // generic driver catch
            return false;
        }
        static bool isStdLibNoise(const std::string& name) {
            if (name.find("std::_Func") == 0) return true;
            if (name.find("std::invoke") == 0) return true;
            if (name.find("std::_") == 0) return true; // Generic std internal
            if (name.find("gpufl::monitor") == 0) return true; // Generic std internal
            return false;
        }
        static std::string simplifySymbol(std::string name) {
            if (name.find("<lambda") != std::string::npos) {
                return "lambda";
            }
            return name;
        }
        static std::string sanitizeStackTrace(const std::string& rawTrace) {
            std::vector<std::string> userStack;
            std::string segment;
            std::stringstream ss(rawTrace);

            // Split by '|'
            while (std::getline(ss, segment, '|')) {

                // STOP: Hit the bottom of the stack (CUDA driver)
                if (isInternalFunction(segment)) {
                    break;
                }

                // SKIP: Filter out noise
                if (segment == "mainCRTStartup" ||
                    segment == "__scrt_common_main" ||
                    segment == "__scrt_common_main_seh" ||
                    segment == "invoke_main" ||
                    isStdLibNoise(segment)) { // <--- NEW CHECK
                    continue;
                    }

                // RENAME: Clean up the name
                std::string cleanName = simplifySymbol(segment); // <--- NEW HELPER

                userStack.push_back(cleanName);
            }

            // Rebuild string
            std::string cleanTrace;
            for (size_t i = 0; i < userStack.size(); ++i) {
                if (i > 0) cleanTrace += "|";
                cleanTrace += userStack[i];
            }

            return cleanTrace.empty() ? "global" : cleanTrace;
        }

        int64_t getTimestampNs();
        int getPid();
        std::string toIso8601Utc();
    }
}
