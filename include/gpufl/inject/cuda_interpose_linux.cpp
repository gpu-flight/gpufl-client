// Linux/LD_PRELOAD CUDA activity-boundary interposition.
//
// Keep this in a separate translation unit from inject_entry.cpp so every
// exported CUDA symbol is compiled against NVIDIA's official CUDA 13 headers.
// That makes the compiler enforce the ABI for opaque handles and 2D/3D
// parameter structures instead of relying on hand-written void* equivalents.

#if !defined(_WIN32)

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <dlfcn.h>

extern "C" {
void GpuFlightWaitAtCudaLaunchBoundary();
void GpuFlightWaitAtCudaSyncBoundary();
void GpuFlightWaitAtCudaMemoryBoundary();
}

namespace {

template <typename Fn>
Fn ResolveNext(const char* symbol) noexcept {
    // RTLD_NEXT skips libgpufl_inject itself. Resolution is performed once per
    // wrapper by a function-local static below, after the readiness wait; no
    // CUDA API is called while the loader is resolving the real symbol.
    return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, symbol));
}

}  // namespace

#define GPUFL_CUDA_INTERPOSE(WAIT, RETURN_TYPE, NAME, PARAMS, ARGS, FAILURE) \
    extern "C" __attribute__((visibility("default")))                      \
    RETURN_TYPE NAME PARAMS {                                               \
        WAIT();                                                             \
        using Function = RETURN_TYPE (*) PARAMS;                            \
        static Function resolved_function = ResolveNext<Function>(#NAME);   \
        return resolved_function ? resolved_function ARGS : FAILURE;        \
    }

// Launch and synchronization boundaries previously lived in inject_entry.cpp
// with ABI-compatible opaque types. Keeping them here means the same official
// headers now validate every interposed CUDA function.
GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaLaunchBoundary, cudaError_t, __cudaLaunchKernel,
    (const void* function_address, dim3 grid_dim, dim3 block_dim, void** args,
     std::size_t shared_mem, cudaStream_t stream),
    (function_address, grid_dim, block_dim, args, shared_mem, stream),
    cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaLaunchBoundary, cudaError_t, cudaLaunchKernel,
    (const void* function_address, dim3 grid_dim, dim3 block_dim, void** args,
     std::size_t shared_mem, cudaStream_t stream),
    (function_address, grid_dim, block_dim, args, shared_mem, stream),
    cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaLaunchBoundary, cudaError_t, cudaLaunchKernelExC,
    (const cudaLaunchConfig_t* config, const void* function_address, void** args),
    (config, function_address, args),
    cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaLaunchBoundary, CUresult, cuLaunchKernel,
    (CUfunction function, unsigned int grid_x, unsigned int grid_y,
     unsigned int grid_z, unsigned int block_x, unsigned int block_y,
     unsigned int block_z, unsigned int shared_mem, CUstream stream,
     void** kernel_params, void** extra),
    (function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem,
     stream, kernel_params, extra),
    CUDA_ERROR_UNKNOWN)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaSyncBoundary, cudaError_t, cudaDeviceSynchronize,
    (), (), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaSyncBoundary, cudaError_t, cudaStreamSynchronize,
    (cudaStream_t stream), (stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaSyncBoundary, CUresult, cuCtxSynchronize,
    (), (), CUDA_ERROR_UNKNOWN)

// CUDA Runtime API: synchronous, asynchronous, pitched, 3D, and peer copies.
GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy,
    (void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind),
    (dst, src, bytes, kind), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpyAsync,
    (void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
     cudaStream_t stream),
    (dst, src, bytes, kind, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpyAsync_ptsz,
    (void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
     cudaStream_t stream),
    (dst, src, bytes, kind, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy2D,
    (void* dst, std::size_t dst_pitch, const void* src, std::size_t src_pitch,
     std::size_t width, std::size_t height, cudaMemcpyKind kind),
    (dst, dst_pitch, src, src_pitch, width, height, kind), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy2DAsync,
    (void* dst, std::size_t dst_pitch, const void* src, std::size_t src_pitch,
     std::size_t width, std::size_t height, cudaMemcpyKind kind,
     cudaStream_t stream),
    (dst, dst_pitch, src, src_pitch, width, height, kind, stream),
    cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy2DAsync_ptsz,
    (void* dst, std::size_t dst_pitch, const void* src, std::size_t src_pitch,
     std::size_t width, std::size_t height, cudaMemcpyKind kind,
     cudaStream_t stream),
    (dst, dst_pitch, src, src_pitch, width, height, kind, stream),
    cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3D,
    (const cudaMemcpy3DParms* params), (params), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3DAsync,
    (const cudaMemcpy3DParms* params, cudaStream_t stream),
    (params, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3DAsync_ptsz,
    (const cudaMemcpy3DParms* params, cudaStream_t stream),
    (params, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpyPeer,
    (void* dst, int dst_device, const void* src, int src_device,
     std::size_t bytes),
    (dst, dst_device, src, src_device, bytes), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpyPeerAsync,
    (void* dst, int dst_device, const void* src, int src_device,
     std::size_t bytes, cudaStream_t stream),
    (dst, dst_device, src, src_device, bytes, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpyPeerAsync_ptsz,
    (void* dst, int dst_device, const void* src, int src_device,
     std::size_t bytes, cudaStream_t stream),
    (dst, dst_device, src, src_device, bytes, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3DPeer,
    (const cudaMemcpy3DPeerParms* params), (params), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3DPeerAsync,
    (const cudaMemcpy3DPeerParms* params, cudaStream_t stream),
    (params, stream), cudaErrorUnknown)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, cudaError_t, cudaMemcpy3DPeerAsync_ptsz,
    (const cudaMemcpy3DPeerParms* params, cudaStream_t stream),
    (params, stream), cudaErrorUnknown)

// CUDA Driver API. The non-stream variants have PTDS aliases; asynchronous
// variants have PTSZ aliases when an application opts into per-thread default
// streams. Export both spellings so the readiness barrier cannot be bypassed
// by that compile-time mode.
GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, CUresult, cuMemcpy,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes),
    (dst, src, bytes), CUDA_ERROR_UNKNOWN)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, CUresult, cuMemcpy_ptds,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes),
    (dst, src, bytes), CUDA_ERROR_UNKNOWN)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, CUresult, cuMemcpyAsync,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream),
    (dst, src, bytes, stream), CUDA_ERROR_UNKNOWN)

GPUFL_CUDA_INTERPOSE(
    GpuFlightWaitAtCudaMemoryBoundary, CUresult, cuMemcpyAsync_ptsz,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream),
    (dst, src, bytes, stream), CUDA_ERROR_UNKNOWN)

#define GPUFL_DRIVER_COPY_SYNC_PAIR(NAME, PARAMS, ARGS)                     \
    GPUFL_CUDA_INTERPOSE(                                                    \
        GpuFlightWaitAtCudaMemoryBoundary, CUresult, NAME, PARAMS, ARGS,    \
        CUDA_ERROR_UNKNOWN)                                                  \
    GPUFL_CUDA_INTERPOSE(                                                    \
        GpuFlightWaitAtCudaMemoryBoundary, CUresult, NAME##_ptds, PARAMS,   \
        ARGS, CUDA_ERROR_UNKNOWN)

#define GPUFL_DRIVER_COPY_ASYNC_PAIR(NAME, PARAMS, ARGS)                    \
    GPUFL_CUDA_INTERPOSE(                                                    \
        GpuFlightWaitAtCudaMemoryBoundary, CUresult, NAME, PARAMS, ARGS,    \
        CUDA_ERROR_UNKNOWN)                                                  \
    GPUFL_CUDA_INTERPOSE(                                                    \
        GpuFlightWaitAtCudaMemoryBoundary, CUresult, NAME##_ptsz, PARAMS,   \
        ARGS, CUDA_ERROR_UNKNOWN)

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpyHtoD_v2,
    (CUdeviceptr dst, const void* src, std::size_t bytes),
    (dst, src, bytes))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpyDtoH_v2,
    (void* dst, CUdeviceptr src, std::size_t bytes),
    (dst, src, bytes))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpyDtoD_v2,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes),
    (dst, src, bytes))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpyHtoDAsync_v2,
    (CUdeviceptr dst, const void* src, std::size_t bytes, CUstream stream),
    (dst, src, bytes, stream))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpyDtoHAsync_v2,
    (void* dst, CUdeviceptr src, std::size_t bytes, CUstream stream),
    (dst, src, bytes, stream))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpyDtoDAsync_v2,
    (CUdeviceptr dst, CUdeviceptr src, std::size_t bytes, CUstream stream),
    (dst, src, bytes, stream))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpy2D_v2,
    (const CUDA_MEMCPY2D* params),
    (params))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpy2DAsync_v2,
    (const CUDA_MEMCPY2D* params, CUstream stream),
    (params, stream))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpy3D_v2,
    (const CUDA_MEMCPY3D* params),
    (params))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpy3DAsync_v2,
    (const CUDA_MEMCPY3D* params, CUstream stream),
    (params, stream))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpyPeer,
    (CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src,
     CUcontext src_context, std::size_t bytes),
    (dst, dst_context, src, src_context, bytes))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpyPeerAsync,
    (CUdeviceptr dst, CUcontext dst_context, CUdeviceptr src,
     CUcontext src_context, std::size_t bytes, CUstream stream),
    (dst, dst_context, src, src_context, bytes, stream))

GPUFL_DRIVER_COPY_SYNC_PAIR(
    cuMemcpy3DPeer,
    (const CUDA_MEMCPY3D_PEER* params),
    (params))

GPUFL_DRIVER_COPY_ASYNC_PAIR(
    cuMemcpy3DPeerAsync,
    (const CUDA_MEMCPY3D_PEER* params, CUstream stream),
    (params, stream))

#undef GPUFL_DRIVER_COPY_ASYNC_PAIR
#undef GPUFL_DRIVER_COPY_SYNC_PAIR
#undef GPUFL_CUDA_INTERPOSE

#endif  // !defined(_WIN32)
