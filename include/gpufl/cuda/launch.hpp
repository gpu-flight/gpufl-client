#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpufl/gpufl.hpp"

#if !defined(__CUDACC__)
  #error "gpufl/cuda/launch.hpp must be compiled with NVCC (__CUDACC__)"
#endif
#define GFL_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...) \
    do { \
        dim3 _gDim(grid); \
        dim3 _bDim(block); \
        \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel); \
        std::string _gStr = gpufl::cuda::dim3ToString(_gDim); \
        std::string _bStr = gpufl::cuda::dim3ToString(_bDim); \
        \
        int _dev; cudaGetDevice(&_dev); \
        cudaDeviceProp _prop; cudaGetDeviceProperties(&_prop, _dev); \
        int _blockSize = _bDim.x * _bDim.y * _bDim.z; \
        int _maxActiveBlocks = 0; \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&_maxActiveBlocks, kernel, _blockSize, sharedMem); \
        \
        float _occupancy = 0.0f; \
        if (_prop.maxThreadsPerMultiProcessor > 0) { \
            int _activeWarps = _maxActiveBlocks * (_blockSize / _prop.warpSize); \
            int _maxWarps = _prop.maxThreadsPerMultiProcessor / _prop.warpSize; \
            _occupancy = (float)_activeWarps / (float)_maxWarps; \
        } \
        \
        { \
            gpufl::cuda::KernelMonitor _monitor(#kernel, tag, _gStr, _bStr, \
                (int)(sharedMem), _attrs.numRegs, _attrs.sharedSizeBytes, \
                _attrs.localSizeBytes, _attrs.constSizeBytes, \
                _occupancy, _maxActiveBlocks); \
            \
            kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
            \
            cudaError_t _err = cudaGetLastError(); \
            if (_err == cudaSuccess) { \
                cudaError_t _syncErr = cudaDeviceSynchronize(); \
                if (_syncErr != cudaSuccess) _err = _syncErr; \
            } \
            _monitor.setError(cudaGetErrorString(_err)); \
        } \
    } while(0)

#define GFL_LAUNCH(kernel, grid, block, sharedMem, stream, ...) \
    GFL_LAUNCH_TAGGED("", kernel, grid, block, sharedMem, stream, __VA_ARGS__)