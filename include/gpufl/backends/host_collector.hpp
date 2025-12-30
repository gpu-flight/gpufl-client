#pragma once
#include "gpufl/core/events.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <fstream>
#include <string>
#include <sstream>
#endif

namespace gpufl {

    class HostCollector {
    public:
        HostCollector() {
            // Initialize previous timestamps for CPU calculation
            sampleCpu();
        }

        HostSample sample() {
            HostSample s;
            s.cpuUtilPercent = sampleCpu();
            sampleRam(s);
            return s;
        }

    private:
#if defined(_WIN32)
        // --- WINDOWS IMPLEMENTATION ---
        uint64_t prevIdleTime_ = 0;
        uint64_t prevKernelTime_ = 0;
        uint64_t prevUserTime_ = 0;

        double sampleCpu() {
            FILETIME idle, kernel, user;
            if (!GetSystemTimes(&idle, &kernel, &user)) return 0.0;

            auto toU64 = [](const FILETIME& ft) {
                return static_cast<uint64_t>(ft.dwLowDateTime) | (static_cast<uint64_t>(ft.dwHighDateTime) << 32);
            };

            const uint64_t curIdle = toU64(idle);
            const uint64_t curKernel = toU64(kernel);
            const uint64_t curUser = toU64(user);

            const uint64_t diffIdle = curIdle - prevIdleTime_;
            const uint64_t diffKernel = curKernel - prevKernelTime_;
            const uint64_t diffUser = curUser - prevUserTime_;

            // On Windows, KernelTime includes IdleTime.
            // Total = (Kernel - Idle) + User + Idle  => Kernel + User
            const uint64_t totalSys = diffKernel + diffUser;

            // However, since Kernel includes Idle, the non-idle kernel time is (Kernel - Idle).
            // Active = (diffKernel - diffIdle) + diffUser
            // But denominator (Total Time passed) is just diffKernel + diffUser

            double percent = 0.0;
            if (totalSys > 0) {
                 // Active part is Total - Idle part
                 // Since Kernel includes Idle, 'totalSys' is the total wall time.
                 // The 'Idle' variable is the idle component of Kernel.
                 const uint64_t active = totalSys - diffIdle;
                 percent = static_cast<double>(active) / static_cast<double>(totalSys) * 100.0;
            }

            prevIdleTime_ = curIdle;
            prevKernelTime_ = curKernel;
            prevUserTime_ = curUser;

            return percent;
        }

        static void sampleRam(HostSample& s) {
            MEMORYSTATUSEX memInfo;
            memInfo.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&memInfo)) {
                s.ramTotalMiB = memInfo.ullTotalPhys / (1024 * 1024);
                s.ramUsedMiB = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);
            }
        }

#else
        // --- LINUX IMPLEMENTATION ---
        struct CpuTicks {
            unsigned long long user=0, nice=0, system=0, idle=0, iowait=0, irq=0, softirq=0, steal=0;
        };
        CpuTicks prev_;

        double sampleCpu() {
            std::ifstream f("/proc/stat");
            if (!f.is_open()) return 0.0;

            std::string line;
            std::getline(f, line); // first line is usually "cpu  ..."
            if (line.substr(0, 3) != "cpu") return 0.0;

            std::istringstream ss(line.substr(4));
            CpuTicks cur;
            ss >> cur.user >> cur.nice >> cur.system >> cur.idle
               >> cur.iowait >> cur.irq >> cur.softirq >> cur.steal;

            unsigned long long prevIdle = prev_.idle + prev_.iowait;
            unsigned long long curIdle  = cur.idle + cur.iowait;

            unsigned long long prevNonIdle = prev_.user + prev_.nice + prev_.system + prev_.irq + prev_.softirq + prev_.steal;
            unsigned long long curNonIdle  = cur.user + cur.nice + cur.system + cur.irq + cur.softirq + cur.steal;

            unsigned long long prevTotal = prevIdle + prevNonIdle;
            unsigned long long curTotal  = curIdle + curNonIdle;

            unsigned long long totalDiff = curTotal - prevTotal;
            unsigned long long idleDiff  = curIdle - prevIdle;

            double percent = 0.0;
            if (totalDiff > 0) {
                percent = (double)(totalDiff - idleDiff) / (double)totalDiff * 100.0;
            }

            prev_ = cur;
            return percent;
        }

        void sampleRam(HostSample& s) {
            struct sysinfo info;
            if (sysinfo(&info) == 0) {
                // sysinfo units can vary (mem_unit), usually 1
                uint64_t total = (uint64_t)info.totalram * info.mem_unit;
                uint64_t free = (uint64_t)info.freeram * info.mem_unit;
                // Buffers/cache are often counted as "used" in raw math but "available" effectively.
                // For simplicity here: Used = Total - Free.
                s.ramTotalMiB = total / (1024 * 1024);
                s.ramUsedMiB = (total - free) / (1024 * 1024);
            }
        }
#endif
    };
}