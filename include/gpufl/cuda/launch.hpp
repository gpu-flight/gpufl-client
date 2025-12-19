#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpufl/gpufl.hpp"

#if !defined(__CUDACC__)
  #error "gpufl/cuda/launch.hpp must be compiled with NVCC (__CUDACC__)"
#endif

#define GFL_LAUNCH(kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _gfl_ts_start = gpufl::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _gfl_err = cudaGetLastError(); \
        cudaError_t _gfl_sync_err = cudaDeviceSynchronize(); \
        if (_gfl_sync_err != cudaSuccess && _gfl_err == cudaSuccess) { \
            _gfl_err = _gfl_sync_err; \
        } \
        int64_t _gfl_ts_end = gpufl::detail::getTimestampNs(); \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel); \
        gpufl::cuda::logKernelEvent(#kernel, _gfl_ts_start, _gfl_ts_end, grid, block, sharedMem, gpufl::cuda::getCudaErrorString(_gfl_err), _attrs); \
    } while(0)

// wraps a single kernel launch with custom tag
#define GFL_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _gfl_ts_start = gpufl::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _gfl_err = cudaGetLastError(); \
        cudaError_t _gfl_sync_err = cudaDeviceSynchronize(); \
        if (_gfl_sync_err != cudaSuccess && _gfl_err == cudaSuccess) { \
            _gfl_err = _gfl_sync_err; \
        } \
        int64_t _gfl_ts_end = gpufl::detail::getTimestampNs(); \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel); \
        gpufl::cuda::logKernelEvent(#kernel, _gfl_ts_start, _gfl_ts_end, grid, block, sharedMem, gpufl::cuda::getCudaErrorString(_gfl_err), _attrs, tag); \
    } while(0)

