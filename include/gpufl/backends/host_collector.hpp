#pragma once
#include "gpufl/core/events.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <sys/sysinfo.h>

#include <fstream>
#include <sstream>
#include <string>
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
        s.cpu_util_percent = sampleCpu();
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
            return static_cast<uint64_t>(ft.dwLowDateTime) |
                   (static_cast<uint64_t>(ft.dwHighDateTime) << 32);
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

        // However, since Kernel includes Idle, the non-idle kernel time is
        // (Kernel - Idle). Active = (diffKernel - diffIdle) + diffUser But
        // denominator (Total Time passed) is just diffKernel + diffUser

        double percent = 0.0;
        if (totalSys > 0) {
            // Active part is Total - Idle part
            // Since Kernel includes Idle, 'totalSys' is the total wall time.
            // The 'Idle' variable is the idle component of Kernel.
            const uint64_t active = totalSys - diffIdle;
            percent = static_cast<double>(active) /
                      static_cast<double>(totalSys) * 100.0;
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
            s.ram_total_mib = memInfo.ullTotalPhys / (1024 * 1024);
            s.ram_used_mib =
                (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);
        }
    }

#elif defined(__APPLE__)
    // --- MACOS IMPLEMENTATION ---
    struct CpuTicks {
        uint64_t user = 0;
        uint64_t system = 0;
        uint64_t idle = 0;
        uint64_t nice = 0;
    };
    CpuTicks prev_;

    static bool readCpuTicks(CpuTicks& ticks) {
        host_cpu_load_info_data_t info{};
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        const kern_return_t rc = host_statistics(
            mach_host_self(), HOST_CPU_LOAD_INFO,
            reinterpret_cast<host_info_t>(&info), &count);
        if (rc != KERN_SUCCESS) return false;

        ticks.user = info.cpu_ticks[CPU_STATE_USER];
        ticks.system = info.cpu_ticks[CPU_STATE_SYSTEM];
        ticks.idle = info.cpu_ticks[CPU_STATE_IDLE];
        ticks.nice = info.cpu_ticks[CPU_STATE_NICE];
        return true;
    }

    double sampleCpu() {
        CpuTicks cur;
        if (!readCpuTicks(cur)) return 0.0;

        const uint64_t prevTotal =
            prev_.user + prev_.system + prev_.idle + prev_.nice;
        const uint64_t curTotal = cur.user + cur.system + cur.idle + cur.nice;
        const uint64_t totalDiff = curTotal - prevTotal;
        const uint64_t idleDiff = cur.idle - prev_.idle;

        double percent = 0.0;
        if (totalDiff > 0) {
            percent = static_cast<double>(totalDiff - idleDiff) /
                      static_cast<double>(totalDiff) * 100.0;
        }

        prev_ = cur;
        return percent;
    }

    static void sampleRam(HostSample& s) {
        uint64_t totalBytes = 0;
        size_t totalSize = sizeof(totalBytes);
        if (sysctlbyname("hw.memsize", &totalBytes, &totalSize, nullptr, 0) == 0) {
            s.ram_total_mib = totalBytes / (1024 * 1024);
        }

        vm_statistics64_data_t vmStats{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        const kern_return_t rc = host_statistics64(
            mach_host_self(), HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&vmStats), &count);
        if (rc == KERN_SUCCESS && s.ram_total_mib > 0) {
            const uint64_t pageSize = static_cast<uint64_t>(getpagesize());
            const uint64_t freeBytes =
                static_cast<uint64_t>(vmStats.free_count) * pageSize;
            const uint64_t totalMiB = s.ram_total_mib;
            const uint64_t freeMiB = freeBytes / (1024 * 1024);
            s.ram_used_mib = totalMiB > freeMiB ? totalMiB - freeMiB : 0;
        }
    }

#else
    // --- LINUX IMPLEMENTATION ---
    struct CpuTicks {
        unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0,
                           irq = 0, softirq = 0, steal = 0;
    };
    CpuTicks prev_;

    double sampleCpu() {
        std::ifstream f("/proc/stat");
        if (!f.is_open()) return 0.0;

        std::string line;
        std::getline(f, line);  // first line is usually "cpu  ..."
        if (line.substr(0, 3) != "cpu") return 0.0;

        std::istringstream ss(line.substr(4));
        CpuTicks cur;
        ss >> cur.user >> cur.nice >> cur.system >> cur.idle >> cur.iowait >>
            cur.irq >> cur.softirq >> cur.steal;

        unsigned long long prevIdle = prev_.idle + prev_.iowait;
        unsigned long long curIdle = cur.idle + cur.iowait;

        unsigned long long prevNonIdle = prev_.user + prev_.nice +
                                         prev_.system + prev_.irq +
                                         prev_.softirq + prev_.steal;
        unsigned long long curNonIdle = cur.user + cur.nice + cur.system +
                                        cur.irq + cur.softirq + cur.steal;

        unsigned long long prevTotal = prevIdle + prevNonIdle;
        unsigned long long curTotal = curIdle + curNonIdle;

        unsigned long long totalDiff = curTotal - prevTotal;
        unsigned long long idleDiff = curIdle - prevIdle;

        double percent = 0.0;
        if (totalDiff > 0) {
            percent =
                (double)(totalDiff - idleDiff) / (double)totalDiff * 100.0;
        }

        prev_ = cur;
        return percent;
    }

    void sampleRam(HostSample& s) {
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            // sysinfo units can vary (mem_unit), usually 1
            const uint64_t total = info.totalram * info.mem_unit;
            const uint64_t free = info.freeram * info.mem_unit;
            // Buffers/cache are often counted as "used" in raw math but
            // "available" effectively. For simplicity here: Used = Total -
            // Free.
            s.ram_total_mib = total / (1024 * 1024);
            s.ram_used_mib = (total - free) / (1024 * 1024);
        }
    }
#endif
};
}  // namespace gpufl
