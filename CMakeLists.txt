cmake_minimum_required(VERSION 3.31)

project(gpufl_client
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Header-only GPU monitoring client library"
)

# -----------------------
# CUDA Architectures (CI Friendly)
# -----------------------
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------
# Options
# -----------------------
option(GPUFL_ENABLE_NVIDIA "Enable NVIDIA backends (CUDA + NVML when available)" ON)
option(GPUFL_ENABLE_AMD    "Enable AMD backends (ROCm when available)" OFF)

option(BUILD_GPUFL_EXAMPLE "Build gpufl example application" ON)
option(BUILD_GPUFL_MONITOR "Build gpufl-monitor standalone daemon" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTING "Build the testing tree." ON)

# -----------------------
# Library target
# -----------------------
add_library(gpufl STATIC)
add_library(gpufl::gpufl ALIAS gpufl)

target_include_directories(gpufl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(gpufl INTERFACE cxx_std_17)

# Enable PIC for static library (required when linking into shared libraries like Python modules)
set_target_properties(gpufl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(gpufl PRIVATE
    include/gpufl/core/dictionary_manager.cpp
    include/gpufl/core/sass_compressor.cpp
    include/gpufl/core/logger/logger.cpp
    include/gpufl/core/logger/log_rotator.cpp
    include/gpufl/core/model/batch_models.cpp
    include/gpufl/core/model/lifecycle_model.cpp
    include/gpufl/core/model/kernel_event_model.cpp
    include/gpufl/core/model/memcpy_event_model.cpp
    include/gpufl/core/model/scope_event_model.cpp
    include/gpufl/core/model/profile_sample_model.cpp
    include/gpufl/core/model/perf_metric_model.cpp
    include/gpufl/core/model/nvtx_marker_model.cpp
    include/gpufl/core/model/system_event_model.cpp
    include/gpufl/core/sampler.cpp
    include/gpufl/core/runtime.cpp
    include/gpufl/core/backend_factory.cpp
    include/gpufl/core/monitor_adapter.cpp
    include/gpufl/core/monitor.cpp
    include/gpufl/core/gpufl.cpp
    include/gpufl/core/common.cpp
    include/gpufl/core/debug_logger.cpp
    include/gpufl/core/stack_trace.cpp
    include/gpufl/core/scope_registry.cpp
    include/gpufl/core/json/json.cpp
    include/gpufl/core/config_file_loader.cpp
    include/gpufl/report/hint_engine.cpp
    include/gpufl/report/text_report.cpp
)

set(GPUFL_HAS_CUDA 0)
set(GPUFL_HAS_NVML 0)
set(GPUFL_HAS_ROCM 0)
set(GPUFL_HAS_ROCM_SMI 0)
set(GPUFL_HAS_HIP 0)
set(GPUFL_HAS_ROCPROFILER_SDK 0)
set(GPUFL_HAS_CUPTI 0)
set(GPUFL_HAS_PERFWORKS 0)
# ZLIB — try system install first, fall back to FetchContent so every platform
# (including Windows CI) always gets compression support and .gz output.
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    message(STATUS "Found system ZLIB: ${ZLIB_LIBRARIES}")
    target_link_libraries(gpufl PRIVATE ZLIB::ZLIB)
else()
    message(STATUS "ZLIB not found on system — fetching via FetchContent")
    include(FetchContent)
    # Suppress zlib's own example / test targets to keep the build clean
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        v1.3.1
    )
    FetchContent_MakeAvailable(zlib)
    # zlib.h lives in the source dir; zconf.h is generated in the binary dir.
    # Add both privately to gpufl — consumers never include zlib headers directly.
    target_link_libraries(gpufl PRIVATE zlibstatic)
    target_include_directories(gpufl PRIVATE
        ${zlib_SOURCE_DIR}
        ${zlib_BINARY_DIR}
    )
    # zlib's own CMakeLists sets INTERFACE_INCLUDE_DIRECTORIES on zlibstatic to
    # build-directory paths, which CMake forbids in install exports.  Clear them:
    # gpufl already propagates the paths privately, so consumers don't need them.
    set_target_properties(zlibstatic PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    # zlibstatic must be in the same export set as gpufl — static lib consumers
    # need to link it transitively, so CMake requires it to be exported too.
    install(TARGETS zlibstatic
        EXPORT   gpufl_clientTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
target_sources(gpufl PRIVATE include/gpufl/core/logger/file_compressor.cpp)


# -----------------------
# Backends
# -----------------------
if(GPUFL_ENABLE_NVIDIA)
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_NVIDIA=1)
else()
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_NVIDIA=0)
endif()

if(GPUFL_ENABLE_AMD)
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_AMD=1)
else()
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_AMD=0)
endif()

if(GPUFL_ENABLE_NVIDIA)
    #
    # CUDA capability: only if CUDA toolkit is available
    #
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(GPUFL_HAS_CUDA 1)
            target_sources(gpufl PRIVATE
                include/gpufl/backends/nvidia/sampler/cupti_sass.cpp
                include/gpufl/backends/nvidia/sampler/cupti_sass.hpp
                include/gpufl/backends/nvidia/cuda_collector.cpp
                include/gpufl/backends/nvidia/cupti_utils.cpp
                include/gpufl/backends/nvidia/resource_handler.cpp
                include/gpufl/backends/nvidia/kernel_launch_handler.cpp
                include/gpufl/backends/nvidia/mem_transfer_handler.cpp
                include/gpufl/backends/nvidia/monitor_adapter_nvidia.cpp
                include/gpufl/backends/nvidia/cupti_backend.cpp
                include/gpufl/backends/nvidia/engine/pc_sampling_engine.cpp
                include/gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.cpp
                include/gpufl/backends/nvidia/engine/sass_metrics_engine.cpp
                include/gpufl/backends/nvidia/engine/range_profiler_engine.cpp)
            target_link_libraries(gpufl PRIVATE CUDA::cudart CUDA::cuda_driver)

            # --------------------------------------------------------
            # CUPTI Support (Added)
            # --------------------------------------------------------
            if (TARGET CUDA::cupti)
                target_link_libraries(gpufl PRIVATE CUDA::cupti)
                set(GPUFL_HAS_CUPTI 1)
                message(STATUS "Found CUPTI via CUDAToolkit target")
            else()
                # Fallback: Manual search if the target is missing
                find_library(CUPTI_LIBRARY NAMES cupti
                    HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
                    "${CUDAToolkit_ROOT}/extras/CUPTI/lib"
                    "$ENV{CUDA_PATH}/extras/CUPTI/lib64"
                )
                find_path(CUPTI_INCLUDE_DIR NAMES cupti.h
                    HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/include"
                    "$ENV{CUDA_PATH}/extras/CUPTI/include"
                )

                if(CUPTI_LIBRARY AND CUPTI_INCLUDE_DIR)
                    target_link_libraries(gpufl PRIVATE ${CUPTI_LIBRARY})
                    target_include_directories(gpufl PRIVATE ${CUPTI_INCLUDE_DIR})
                    set(GPUFL_HAS_CUPTI 1)
                    message(STATUS "Found CUPTI manually: ${CUPTI_LIBRARY}")
                endif()
            endif()

            # --------------------------------------------------------
            # NVTX (NVIDIA Tools Extension) — lets GFL_SCOPE emit ranges
            # visible to Nsight Systems and captured via CUPTI marker
            # activity, plus lets the PyTorch integration (gpufl.torch)
            # bridge dispatch events into the profiling pipeline.
            # Shipped with the CUDA toolkit; no new external dependency.
            # --------------------------------------------------------
            # NVTX detection strategy (robust across CUDA 11/12/13):
            #   1. CUDA 11/12: `CUDA::nvToolsExt` imported target exists and
            #      links the legacy nvToolsExt DLL/SO. Use it.
            #   2. CUDA 13+: NVTX is header-only under `include/nvtx3/`.
            #      The CUDAToolkit include dir is propagated via CUDA::cudart
            #      (already linked above), so the header is findable — just
            #      define GPUFL_HAS_NVTX=1 and let `#include <nvtx3/nvToolsExt.h>`
            #      resolve through cudart's transitive include.
            #   3. Fallback find_path as a last resort for unusual layouts.
            if (TARGET CUDA::nvToolsExt)
                target_link_libraries(gpufl PRIVATE CUDA::nvToolsExt)
                target_compile_definitions(gpufl PRIVATE GPUFL_HAS_NVTX=1)
                message(STATUS "Found NVTX via CUDAToolkit target (CUDA::nvToolsExt)")
            else()
                # Force-fresh the search (don't reuse cached NOTFOUND from a prior configure)
                unset(NVTX3_INCLUDE_DIR CACHE)
                find_path(NVTX3_INCLUDE_DIR NAMES nvtx3/nvToolsExt.h
                    HINTS "${CUDAToolkit_INCLUDE_DIRS}"
                          "${CUDAToolkit_ROOT}/include"
                          "$ENV{CUDA_PATH}/include"
                    NO_DEFAULT_PATH
                )
                if(NVTX3_INCLUDE_DIR)
                    target_include_directories(gpufl PRIVATE ${NVTX3_INCLUDE_DIR})
                    target_compile_definitions(gpufl PRIVATE GPUFL_HAS_NVTX=1)
                    message(STATUS "Found NVTX headers at ${NVTX3_INCLUDE_DIR}")
                else()
                    # Last-resort: if CUDAToolkit is present at all, the header
                    # is in its include dir via CUDA::cudart's transitive
                    # propagation. Just define the macro; include will resolve.
                    target_compile_definitions(gpufl PRIVATE GPUFL_HAS_NVTX=1)
                    message(STATUS "NVTX: relying on CUDA::cudart transitive include for <nvtx3/nvToolsExt.h>")
                endif()
            endif()

            # --------------------------------------------------------
            # NVPERF Support (for GFL_PERF_SCOPE hardware counters)
            # --------------------------------------------------------
            find_library(NVPERF_HOST_LIBRARY NAMES nvperf_host nvperf_host_static
                HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
                      "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib"
                      "$ENV{CUDA_PATH}/extras/CUPTI/lib64"
                      "$ENV{CUDA_PATH}/targets/x86_64-linux/lib"
            )
            find_library(NVPERF_TARGET_LIBRARY NAMES nvperf_target
                HINTS "${CUDAToolkit_ROOT}/extras/CUPTI/lib64"
                      "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib"
                      "$ENV{CUDA_PATH}/extras/CUPTI/lib64"
                      "$ENV{CUDA_PATH}/targets/x86_64-linux/lib"
            )
            if(NVPERF_HOST_LIBRARY AND NVPERF_TARGET_LIBRARY)
                set(GPUFL_HAS_PERFWORKS 1)
                target_link_libraries(gpufl PRIVATE
                    ${NVPERF_HOST_LIBRARY} ${NVPERF_TARGET_LIBRARY})
                message(STATUS "Found NVPERF: ${NVPERF_HOST_LIBRARY}")
            else()
                set(GPUFL_HAS_PERFWORKS 0)
                message(WARNING "NVPERF not found — GFL_PERF_SCOPE disabled at runtime")
            endif()
        endif()
    endif()
    #
    # NVML capability:
    # - On Linux, link libnvidia-ml if present.
    # - On Windows, many projects LoadLibrary/GetProcAddress at runtime.
    #   If you do runtime loading, treat NVML as "capable" without link-time lib.
    #

    if(WIN32)
        # nvml.dll usually comes from the NVIDIA driver (NVSMI), not CUDA toolkit.
        find_path(NVML_DLL_DIR NAMES nvml.dll
            PATHS
            "$ENV{ProgramFiles}/NVIDIA Corporation/NVSMI"
            "$ENV{SystemRoot}/System32"
        )

        if(NVML_DLL_DIR)
            set(GPUFL_HAS_NVML 1)

            # If you still want link-time import lib, only do it when DLL exists:
            find_library(NVML_LIBRARY NAMES nvml nvidia-ml
                PATHS "$ENV{CUDA_PATH}/lib/x64"
                PATH_SUFFIXES lib lib/x64
            )
            if(NVML_LIBRARY)
                target_link_libraries(gpufl PRIVATE ${NVML_LIBRARY})
            endif()
        endif()
    else()
        find_library(NVML_LIBRARY NAMES nvidia-ml
                HINTS
                "${CUDAToolkit_LIBRARY_ROOT}/stubs"
                "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/stubs"
                "${CUDAToolkit_ROOT}/lib64/stubs"
                "$ENV{CUDA_PATH}/targets/x86_64-linux/lib/stubs"
                "/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs"
                "/usr/lib/wsl/lib"  # Common location for WSL2 users
        )

        if(NVML_LIBRARY)
            set(GPUFL_HAS_NVML 1)
            target_link_libraries(gpufl PRIVATE ${NVML_LIBRARY})
            message(STATUS "Found NVML (Linux): ${NVML_LIBRARY}")
        else()
            message(WARNING "NVML not found! GPUFL_HAS_NVML will be 0. (Check CUDA Toolkit 'stubs' folder)")
        endif()
    endif()

    if(WIN32)
        # dbghelp: stack traces. pdh: PDH GPU/CPU perf counters.
        # psapi: EnumProcessModules for runtime NVTX injection path discovery.
        target_link_libraries(gpufl PRIVATE dbghelp pdh psapi)
    endif()

    #
    # Only compile NVML collector if NVML is actually available
    #
    if(GPUFL_HAS_NVML)
        target_sources(gpufl PRIVATE
            include/gpufl/backends/nvidia/nvml_collector.cpp
        )
    endif()

    # -----------------------
    # CUDA Example (only when CUDA is available)
    # -----------------------
    if(BUILD_GPUFL_EXAMPLE AND GPUFL_HAS_CUDA AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        add_subdirectory(example/cuda)
    endif()

endif()


# -----------------------
# AMD backends
# -----------------------
if(GPUFL_ENABLE_AMD)
    find_package(rocm_smi CONFIG QUIET HINTS /opt/rocm)
    if(rocm_smi_FOUND OR TARGET rocm_smi64)
        set(GPUFL_HAS_ROCM 1)
        set(GPUFL_HAS_ROCM_SMI 1)
        if(TARGET rocm_smi64)
            target_link_libraries(gpufl PRIVATE rocm_smi64)
        elseif(ROCM_SMI_LIBRARY)
            target_link_libraries(gpufl PRIVATE ${ROCM_SMI_LIBRARY})
        endif()
        if(DEFINED rocm_smi_INCLUDE_DIR)
            target_include_directories(gpufl PRIVATE ${rocm_smi_INCLUDE_DIR})
        elseif(DEFINED ROCM_SMI_INCLUDE_DIRS)
            target_include_directories(gpufl PRIVATE ${ROCM_SMI_INCLUDE_DIRS})
        endif()
        message(STATUS "Found ROCm SMI support")
    else()
        find_library(ROCM_SMI_LIBRARY NAMES rocm_smi64 librocm_smi64.so
            HINTS /opt/rocm/lib /opt/rocm/lib64)
        find_path(ROCM_SMI_INCLUDE_DIR NAMES rocm_smi/rocm_smi.h
            HINTS /opt/rocm/include)
        if(ROCM_SMI_LIBRARY AND ROCM_SMI_INCLUDE_DIR)
            set(GPUFL_HAS_ROCM 1)
            set(GPUFL_HAS_ROCM_SMI 1)
            target_link_libraries(gpufl PRIVATE ${ROCM_SMI_LIBRARY})
            target_include_directories(gpufl PRIVATE ${ROCM_SMI_INCLUDE_DIR})
            message(STATUS "Found ROCm SMI manually: ${ROCM_SMI_LIBRARY}")
        else()
            message(WARNING "ROCm SMI not found! GPUFL_HAS_ROCM will be 0.")
        endif()
    endif()

    find_package(hip CONFIG QUIET HINTS /opt/rocm)
    if(hip_FOUND OR TARGET hip::host)
        set(GPUFL_HAS_HIP 1)
        if(TARGET hip::host)
            target_link_libraries(gpufl PRIVATE hip::host)
        endif()
        message(STATUS "Found HIP host runtime support")
    else()
        find_library(AMD_HIP_LIBRARY NAMES amdhip64
            HINTS /opt/rocm/lib /opt/rocm/lib64 /lib/x86_64-linux-gnu)
        find_path(AMD_HIP_INCLUDE_DIR NAMES hip/hip_runtime_api.h
            HINTS /opt/rocm/include)
        if(AMD_HIP_LIBRARY AND AMD_HIP_INCLUDE_DIR)
            set(GPUFL_HAS_HIP 1)
            target_link_libraries(gpufl PRIVATE ${AMD_HIP_LIBRARY})
            target_include_directories(gpufl PRIVATE ${AMD_HIP_INCLUDE_DIR})
            message(STATUS "Found HIP manually: ${AMD_HIP_LIBRARY}")
        else()
            message(WARNING "HIP runtime not found! AMD static inventory will be unavailable.")
        endif()
    endif()

    find_package(rocprofiler-sdk QUIET CONFIG HINTS /opt/rocm)
    if(TARGET rocprofiler-sdk::rocprofiler-sdk)
        set(GPUFL_HAS_ROCPROFILER_SDK 1)
        target_link_libraries(gpufl PRIVATE rocprofiler-sdk::rocprofiler-sdk)
        target_sources(gpufl PRIVATE
            include/gpufl/backends/amd/monitor_adapter_amd.cpp
            include/gpufl/backends/amd/rocprofiler_backend.cpp
            include/gpufl/backends/amd/engine/dispatch_counter_engine.cpp
        )
        message(STATUS "Found ROCprofiler-SDK support")
    else()
        message(STATUS "ROCprofiler-SDK not found; AMD kernel tracing disabled")
    endif()

    if(GPUFL_HAS_ROCM_SMI OR GPUFL_HAS_HIP)
        target_sources(gpufl PRIVATE
            include/gpufl/backends/amd/rocm_collector.cpp
        )
    endif()

    if(BUILD_GPUFL_EXAMPLE AND GPUFL_HAS_HIP AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        add_subdirectory(example/amd)
    endif()
endif()

target_compile_definitions(gpufl PUBLIC
    GPUFL_HAS_CUDA=${GPUFL_HAS_CUDA}
    GPUFL_HAS_NVML=${GPUFL_HAS_NVML}
    GPUFL_HAS_CUPTI=${GPUFL_HAS_CUPTI}
    GPUFL_HAS_PERFWORKS=${GPUFL_HAS_PERFWORKS}
    GPUFL_HAS_ROCM=${GPUFL_HAS_ROCM}
    GPUFL_HAS_ROCM_SMI=${GPUFL_HAS_ROCM_SMI}
    GPUFL_HAS_HIP=${GPUFL_HAS_HIP}
    GPUFL_HAS_ROCPROFILER_SDK=${GPUFL_HAS_ROCPROFILER_SDK}
)

# ---------------------------------------------------------------------------
# gpufl_copy_runtime_dlls(<target>)
#
# On Windows, CUPTI and NVPERF DLLs are not in the system PATH.
# Without these DLLs the exe silently fails to start (no error message).
# Call this function for every executable target that links against gpufl.
#
# Usage (in consumer's CMakeLists.txt):
#   gpufl_copy_runtime_dlls(my_app)
# ---------------------------------------------------------------------------
function(gpufl_copy_runtime_dlls TARGET_NAME)
    if (NOT WIN32)
        return()
    endif()

    set(_SEARCH_DIRS "")
    if (CUDAToolkit_ROOT)
        list(APPEND _SEARCH_DIRS "${CUDAToolkit_ROOT}/extras/CUPTI/lib64")
    endif()
    if (DEFINED ENV{CUDA_PATH})
        file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/extras/CUPTI/lib64" _ENV_DIR)
        list(APPEND _SEARCH_DIRS "${_ENV_DIR}")
    endif()

    foreach(_DIR ${_SEARCH_DIRS})
        file(GLOB _CUPTI "${_DIR}/cupti64*.dll")
        if (_CUPTI)
            file(GLOB _ALL_DLLS
                "${_DIR}/cupti64*.dll"
                "${_DIR}/nvperf_host*.dll"
                "${_DIR}/nvperf_target*.dll")
            foreach(_DLL ${_ALL_DLLS})
                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_DLL}" "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                    COMMENT "gpufl: copying ${_DLL}")
            endforeach()
            return()
        endif()
    endforeach()

    message(WARNING
        "gpufl: Could not find CUPTI DLLs to copy for target '${TARGET_NAME}'.\n"
        "  The executable may fail to start on Windows without cupti64_*.dll.\n"
        "  Searched: ${_SEARCH_DIRS}\n"
        "  Set CUDA_PATH or ensure CUDAToolkit is findable.")
endfunction()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_GPUFL_MONITOR AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(daemon/monitor)
endif()

if(BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_gpufl_client python/bindings.cpp)

    target_link_libraries(_gpufl_client PRIVATE gpufl::gpufl)

    # If CUDA is available, link it to the Python module
    if(GPUFL_HAS_CUDA)
        target_link_libraries(_gpufl_client PRIVATE CUDA::cudart)
    endif()

    install(TARGETS _gpufl_client DESTINATION gpufl)
endif()



# -----------------------
# Install
# -----------------------
include(GNUInstallDirs)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS gpufl
    EXPORT gpufl_clientTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT gpufl_clientTargets
    FILE gpufl_clientTargets.cmake
    NAMESPACE gpufl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpufl_client
)
