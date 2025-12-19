#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpufl/gpufl.hpp"

#if !defined(__CUDACC__)
  #error "gpufl/cuda/launch.hpp must be compiled with NVCC (__CUDACC__)"
#endif
#define GFL_LAUNCH(kernel, grid, block, sharedMem, stream, ...)                      \
    do {                                                                             \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel);                 \
        std::string _gStr = gpufl::cuda::KernelMonitor::dim3ToString(grid);          \
        std::string _bStr = gpufl::cuda::KernelMonitor::dim3ToString(block);         \
        gpufl::cuda::KernelMonitor _monitor(                                         \
            #kernel,                                                                 \
            "",                                                                      \
            _gStr,                                                                   \
            _bStr,                                                                   \
            (sharedMem),                                                             \
            _attrs.numRegs,                                                          \
            _attrs.sharedSizeBytes,                                                  \
            _attrs.localSizeBytes,                                                   \
            _attrs.constSizeBytes                                                    \
        );                                                                           \
                                                                                     \
        kernel<<<(grid), (block), (sharedMem), (stream)>>>(__VA_ARGS__);             \
                                                                                     \
        cudaError_t _err = cudaGetLastError();                                       \
        if (_err == cudaSuccess) {                                                   \
            cudaError_t _syncErr = cudaDeviceSynchronize();                          \
            if (_syncErr != cudaSuccess) _err = _syncErr;                            \
        }                                                                            \
        _monitor.setError(cudaGetErrorString(_err));                                 \
    } while (0)

#define GFL_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...)          \
    do {                                                                             \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel);                 \
        std::string _gStr = gpufl::cuda::KernelMonitor::dim3ToString(grid);          \
        std::string _bStr = gpufl::cuda::KernelMonitor::dim3ToString(block);         \
        gpufl::cuda::KernelMonitor _monitor(                                         \
            #kernel,                                                                 \
            (tag),                                                                   \
            _gStr,                                                                   \
            _bStr,                                                                   \
            (sharedMem),                                                             \
            _attrs.numRegs,                                                          \
            _attrs.sharedSizeBytes,                                                  \
            _attrs.localSizeBytes,                                                   \
            _attrs.constSizeBytes                                                    \
        );                                                                           \
                                                                                     \
        kernel<<<(grid), (block), (sharedMem), (stream)>>>(__VA_ARGS__);             \
                                                                                     \
        cudaError_t _err = cudaGetLastError();                                       \
        if (_err == cudaSuccess) {                                                   \
            cudaError_t _syncErr = cudaDeviceSynchronize();                          \
            if (_syncErr != cudaSuccess) _err = _syncErr;                            \
        }                                                                            \
        _monitor.setError(cudaGetErrorString(_err));                                 \
    } while (0)
