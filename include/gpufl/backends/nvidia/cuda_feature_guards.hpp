#pragma once

// Centralized CUDA-toolkit feature gates.
//
// CUPTI's callback-id enumerators and the `*_params` structs we cast to only
// exist when the CUDA headers we compile against are new enough. Code that
// references them therefore has to be wrapped in a preprocessor guard — a
// runtime check is impossible because the symbol simply isn't declared on
// older toolkits.
//
// Rather than scatter `#if defined(CUDART_VERSION) && CUDART_VERSION >= NNNNN`
// (which encodes a magic version number but not *why*) across every call
// site, we resolve each toolkit threshold to a named feature macro here, once.
// Call sites then read as intent: `#if GPUFL_HAS_EXTENSIBLE_LAUNCH`. If NVIDIA
// ever shifts a threshold, there's a single line to change.
//
// CUDART_VERSION is defined by <cuda_runtime_api.h>; include it so this header
// is self-contained no matter the include order at the call site.
#include <cuda_runtime_api.h>

// Cooperative-group launches: cuLaunchCooperativeKernel /
// cudaLaunchCooperativeKernel (+ MultiDevice, + _ptsz). Used by NCCL
// collectives and any kernel that grid-syncs via cooperative_groups.
// Public since CUDA 9.0.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#  define GPUFL_HAS_COOPERATIVE_LAUNCH 1
#else
#  define GPUFL_HAS_COOPERATIVE_LAUNCH 0
#endif

// Extensible launch: cuLaunchKernelEx / cudaLaunchKernelExC (+ _ptsz). This is
// the path cuda.core takes — and therefore modern numba-cuda (cuda.core >=
// 1.0), CUTLASS, and any thread-block-cluster kernel — because it carries a
// LaunchConfig with launch attributes. Public since CUDA 11.6.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11060
#  define GPUFL_HAS_EXTENSIBLE_LAUNCH 1
#else
#  define GPUFL_HAS_EXTENSIBLE_LAUNCH 0
#endif
