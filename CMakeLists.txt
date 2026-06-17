cmake_minimum_required(VERSION 3.31)

project(gpufl_client
    VERSION 1.3.0
    LANGUAGES CXX
    DESCRIPTION "Header-only GPU monitoring client library"
)

# Pre-release suffix appended to GPUFL_CLIENT_VERSION below. PEP 440
# pre-release tokens (`rc1`, `a1`, `b1`, …) aren't valid in CMake's
# `project(... VERSION ...)`, so we layer them on top here. Final releases
# leave this empty.
set(GPUFL_VERSION_SUFFIX "rc2")

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
if(APPLE)
    set(GPUFL_ENABLE_NVIDIA_DEFAULT OFF)
    set(GPUFL_ENABLE_METAL_DEFAULT ON)
else()
    set(GPUFL_ENABLE_NVIDIA_DEFAULT ON)
    set(GPUFL_ENABLE_METAL_DEFAULT OFF)
endif()

option(GPUFL_ENABLE_NVIDIA "Enable NVIDIA backends (CUDA + NVML when available)" ${GPUFL_ENABLE_NVIDIA_DEFAULT})
option(GPUFL_ENABLE_AMD    "Enable AMD backends (ROCm when available)" OFF)
option(GPUFL_ENABLE_METAL  "Enable Apple Metal backend (macOS basic monitoring)" ${GPUFL_ENABLE_METAL_DEFAULT})
option(GPUFL_ENABLE_AMD_ROCPROFILER "Enable AMD rocprofiler-sdk tracing backend when available" ON)
# OFF builds the configuration release wheels ship (manylinux has no nvperf),
# so CI can catch #if GPUFL_HAS_PERFWORKS mistakes.
option(GPUFL_ENABLE_PERFWORKS "Link PerfWorks (NVPERF) for hardware-counter scopes" ON)

option(BUILD_GPUFL_EXAMPLE "Build gpufl example application" ON)
option(BUILD_GPUFL_MONITOR "Build gpufl-monitor standalone daemon" ON)
# Injection-mode launcher + shared library - Linux + NVIDIA only for v1.
# Disabled by default off-Linux because LD_PRELOAD / __attribute__((constructor))
# / fork+execvp don't apply there.
if(UNIX AND NOT APPLE)
    option(BUILD_GPUFL_LAUNCHER "Build gpufl launcher binary (gpufl trace)" ON)
    option(BUILD_GPUFL_INJECT   "Build libgpufl_inject.so shared library" ON)
else()
    option(BUILD_GPUFL_LAUNCHER "Build gpufl launcher binary (gpufl trace)" OFF)
    option(BUILD_GPUFL_INJECT   "Build libgpufl_inject.so shared library" OFF)
endif()
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTING "Build the testing tree." ON)

# This example is intentionally independent from CUDA, HIP, and gpufl.  It is
# useful as a small CPU-only rendering workload on machines without a GPU.
if(BUILD_GPUFL_EXAMPLE AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(example/cpu)
endif()

# -----------------------
# Library target
# -----------------------
add_library(gpufl STATIC)
add_library(gpufl::gpufl ALIAS gpufl)

# One counter registry per PROCESS, not per module. `gpufl` is static and is
# linked separately into gpufl_inject, the Python extension and any host
# application, so without this each would hold its own registry - a profiled
# target would tick one and the injected evaluator would read another. Only a C
# ABI crosses this boundary; see include/gpufl/abi/gpufl_counter_abi.h.
add_library(gpufl_counter_runtime SHARED
    runtime/counter_runtime.cpp
    include/gpufl/core/counter_registry.cpp
)
target_include_directories(gpufl_counter_runtime PUBLIC include)

# Installed unconditionally, NOT only with the Python bindings. A launcher-only
# install that omits it leaves the injected evaluator and the target application
# each falling back to their own in-process registry - the exact split this
# library exists to prevent, and one that shows up as a counter reading Missing
# forever rather than as a missing file.
install(TARGETS gpufl_counter_runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
set_target_properties(gpufl_counter_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Counters WITHOUT the profiler. What an application links to tick counters a
# rule can watch (`gpufl::counter("tokens").add(n)`), while the profiling
# itself arrives from outside via `gpufl trace`. Linking gpufl::gpufl for two
# lines of instrumentation reads as "I am embedding the SDK" - and the target
# name is most of the answer to that confusion; nothing here starts a session,
# and none of the SDK's dependencies (CUPTI, NVML, zlib, OpenSSL, the
# uploader) are in it.
#
# `gpufl` layers ON TOP of this rather than compiling the same sources again:
# two archives both defining the registry is a duplicate-symbol error for
# anyone who links both.
add_library(gpufl_counters STATIC
    include/gpufl/core/counter_registry.cpp
    include/gpufl/core/counter_api.cpp
    include/gpufl/core/counter_provider.cpp
    include/gpufl/core/debug_logger.cpp
)
add_library(gpufl::counters ALIAS gpufl_counters)
# The ALIAS above only exists in this build tree. The INSTALLED name comes
# from EXPORT_NAME plus the export namespace, and without this the package
# would ship the target as gpufl::gpufl_counters while every document says
# gpufl::counters.
set_target_properties(gpufl_counters PROPERTIES EXPORT_NAME counters)
target_include_directories(gpufl_counters
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(gpufl_counters INTERFACE cxx_std_17)
# The provider dlopens the shared runtime and the registry uses std::mutex;
# consumers must not have to know either. Threads::Threads rather than a raw
# `pthread`: the raw name would be exported verbatim into the targets file and
# bypass whatever the consumer's toolchain says threading means there.
find_package(Threads REQUIRED)
target_link_libraries(gpufl_counters PUBLIC Threads::Threads)
if(UNIX)
    target_link_libraries(gpufl_counters PUBLIC ${CMAKE_DL_LIBS})
endif()
set_target_properties(gpufl_counters PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gpufl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(gpufl INTERFACE cxx_std_17)

# Stamp the CMake project version into the C++ binary so the agent can
# send it as `User-Agent: gpufl/<v>` and `X-GpuFlight-Client-Version`.
# Single source of truth = the project() VERSION line at the top of
# this file. See include/gpufl/core/version.hpp.
#
# PUBLIC (not PRIVATE) so consumers of gpufl - gpufl_tests, daemon/monitor,
# example apps, and any downstream library user - see the same macro
# value when compiling against gpufl/core/version.hpp. With PRIVATE, the
# library's TUs see "0.1.0" but consumer TUs fall back to "0.0.0-dev",
# causing an ODR-style runtime mismatch (the constexpr inline char*
# resolves to different values in different TUs). Release builds
# inline the literal at call sites and the mismatch becomes visible
# as comparison failures (e.g. test asserting on User-Agent).
target_compile_definitions(gpufl PUBLIC
    GPUFL_CLIENT_VERSION="${PROJECT_VERSION}${GPUFL_VERSION_SUFFIX}"
)

# Enable PIC for static library (required when linking into shared libraries like Python modules)
set_target_properties(gpufl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# PUBLIC: consumers of the full SDK get the counter layer through this link,
# and the counter sources live in gpufl_counters ONLY - see that target for
# why they must not be compiled into both archives.
target_link_libraries(gpufl PUBLIC gpufl_counters)

target_sources(gpufl PRIVATE
    include/gpufl/backends/amd/amd_profiling_policy.cpp
    include/gpufl/backends/amd/amd_capture_capabilities.cpp
    include/gpufl/backends/amd/amd_trace_policy.cpp
    include/gpufl/core/dictionary_manager.cpp
    include/gpufl/core/source_capture_policy.cpp
    include/gpufl/core/sass_compressor.cpp
    include/gpufl/core/logger/logger.cpp
    include/gpufl/core/logger/lifecycle_control_journal.cpp
    include/gpufl/core/logger/log_rotator.cpp
    include/gpufl/core/logger/log_salvage.cpp
    include/gpufl/core/logger/session_ownership.cpp
    include/gpufl/core/logger/window_metadata.cpp
    include/gpufl/core/logger/file_log_sink.cpp
    include/gpufl/upload/upload_logs.cpp
    include/gpufl/core/host_info.cpp
    include/gpufl/core/remote_config.cpp
    include/gpufl/core/model/batch_models.cpp
    include/gpufl/core/model/lifecycle_model.cpp
    include/gpufl/core/model/kernel_event_model.cpp
    include/gpufl/core/model/memcpy_event_model.cpp
    include/gpufl/core/model/scope_event_model.cpp
    include/gpufl/core/model/profile_sample_model.cpp
    include/gpufl/core/model/perf_metric_model.cpp
    include/gpufl/core/model/nvtx_marker_model.cpp
    include/gpufl/core/model/synchronization_event_model.cpp
    include/gpufl/core/model/memory_alloc_event_model.cpp
    include/gpufl/core/model/graph_launch_event_model.cpp
    include/gpufl/core/model/graph_node_definition_model.cpp
    include/gpufl/core/model/system_event_model.cpp
    include/gpufl/core/model/deep_window_model.cpp
    include/gpufl/core/deep_window.cpp
    include/gpufl/core/sampler.cpp
    include/gpufl/core/runtime.cpp
    include/gpufl/core/segment_coordinator.cpp
    include/gpufl/core/segment_runtime.cpp
    include/gpufl/core/backend_factory.cpp
    include/gpufl/core/monitor_adapter.cpp
    include/gpufl/core/nvtx_counters.cpp
    include/gpufl/core/metric_id.cpp
    include/gpufl/core/deep_window_rule.cpp
    include/gpufl/core/deep_window_rules.cpp
    include/gpufl/core/metric_registry.cpp
    include/gpufl/core/monitor_batch_manager.cpp
    include/gpufl/core/monitor_record_builders.cpp
    include/gpufl/core/monitor.cpp
    include/gpufl/core/client_startup.cpp
    include/gpufl/core/monitor_configuration.cpp
    include/gpufl/core/gpufl.cpp
    include/gpufl/core/session_bootstrap.cpp
    include/gpufl/core/startup_configuration.cpp
    include/gpufl/core/common.cpp
    include/gpufl/core/stack_trace.cpp
    include/gpufl/core/itanium_demangle.cpp
    include/gpufl/core/scope_registry.cpp
    include/gpufl/core/json/json.cpp
    include/gpufl/core/config_file_loader.cpp
    include/gpufl/report/hint_engine.cpp
    include/gpufl/report/text_report.cpp
)

set(GPUFL_HAS_CUDA 0)
set(GPUFL_HAS_NVML 0)
set(GPUFL_HAS_NVAPI 0)
set(GPUFL_HAS_ROCM 0)
set(GPUFL_HAS_ROCM_SMI 0)
set(GPUFL_HAS_HIP 0)
set(GPUFL_HAS_ROCPROFILER_SDK 0)
set(GPUFL_HAS_METAL 0)
set(GPUFL_HAS_CUPTI 0)
set(GPUFL_HAS_PERFWORKS 0)
# ZLIB - try system install first, fall back to FetchContent so every platform
# (including Windows CI) always gets compression support and .gz output.
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    message(STATUS "Found system ZLIB: ${ZLIB_LIBRARIES}")
    target_link_libraries(gpufl PRIVATE ZLIB::ZLIB)
else()
    message(STATUS "ZLIB not found on system - fetching via FetchContent")
    include(FetchContent)
    # Suppress zlib's own example / test targets to keep the build clean
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        v1.3.1
    )
    # zlib's own install rules cache ABSOLUTE destinations at configure time
    # (INSTALL_LIB_DIR = ${CMAKE_INSTALL_PREFIX}/lib), so they ignore
    # `cmake --install --prefix` and try to write into Program Files - which is
    # how the install-tree consumer test failed before this. Skip them all; the
    # archive is installed below with a relative destination that the prefix
    # override can move.
    set(SKIP_INSTALL_ALL ON)
    FetchContent_MakeAvailable(zlib)
    unset(SKIP_INSTALL_ALL)
    # zlib.h lives in the source dir; zconf.h is generated in the binary dir.
    # Add both privately to gpufl - consumers never include zlib headers directly.
    target_link_libraries(gpufl PRIVATE zlibstatic)
    target_include_directories(gpufl PRIVATE
        ${zlib_SOURCE_DIR}
        ${zlib_BINARY_DIR}
    )
    # zlib's own CMakeLists sets INTERFACE_INCLUDE_DIRECTORIES on zlibstatic to
    # build-directory paths, which CMake forbids in install exports.  Clear them:
    # gpufl already propagates the paths privately, so consumers don't need them.
    set_target_properties(zlibstatic PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    # zlibstatic must be in the same export set as gpufl - static lib consumers
    # need to link it transitively, so CMake requires it to be exported too.
    install(TARGETS zlibstatic
        EXPORT   gpufl_clientTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
target_sources(gpufl PRIVATE include/gpufl/core/logger/file_compressor.cpp)


# -----------------------
# cpp-httplib - HTTP client used by gpufl::uploadLogs (deferred upload
# of session NDJSON files to the backend, called after gpufl::shutdown).
# Also used by remote_config.cpp for the post-init version probe.
#
# Single-header library. Fetched once via FetchContent so every build gets
# the same version regardless of the host system. HTTPS support is gated
# on OpenSSL availability (optional) - without OpenSSL we compile HTTP-only
# and log a warning so the user knows HTTPS endpoints will fail.
# -----------------------
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.5
)
# cpp-httplib's CMakeLists defines build flags for its own tests / examples.
# Turn them off so we only build the header-only interface target.
set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
# HTTPLIB_INSTALL stays ON, deliberately: gpufl's link interface records
# httplib::httplib, and install(EXPORT gpufl_clientTargets) refuses to
# generate unless that target is in SOME export set - httplib's own is what
# satisfies the check. Turning it off fails the whole configure. The price is
# cpp-httplib's headers and package files landing in the install prefix; the
# fix that removes them is reworking the SDK export, not this switch.
set(HTTPLIB_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib)

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "Found OpenSSL: ${OPENSSL_VERSION} - HTTPS upload enabled")
    target_compile_definitions(gpufl PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=1)
    target_link_libraries(gpufl PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    set(GPUFL_HTTPLIB_TLS 1)
else()
    message(WARNING
        "OpenSSL not found - gpufl::uploadLogs will support HTTP only. "
        "Pointing backend_url at an https:// endpoint will fail to verify. "
        "Install OpenSSL (apt: libssl-dev, vcpkg: openssl, brew: openssl) "
        "to enable TLS.")
    set(GPUFL_HTTPLIB_TLS 0)
endif()
target_compile_definitions(gpufl PRIVATE GPUFL_HTTPLIB_TLS=${GPUFL_HTTPLIB_TLS})
# Enable cpp-httplib's gzip path. PUBLIC so any consumer that includes
# httplib.h via gpufl's headers (notably the test target's embedded
# httplib::Server) gets the same wire-format support - without this,
# server::Post handlers return 415 on incoming Content-Encoding: gzip
# requests, which is exactly what uploadLogs sends in v1.2+.
#
# In production, the Spring Boot backend handles gzip natively, so
# this define affects the test target more than the client. We still
# define it on gpufl PUBLIC because:
#   - It enables future use of httplib::Client::set_compress(true)
#     to compress the body in-place instead of our manual gzipString
#     (no current need, but a free optimization if we ever want it).
#   - It propagates to tests, which is the immediate motivation.
#
# ZLIB is already linked above - file_compressor.cpp uses it to gzip
# rotated NDJSON files, and upload_logs.cpp uses it both to read those
# files back and to gzip outgoing stream-chunks.
target_compile_definitions(gpufl PUBLIC CPPHTTPLIB_ZLIB_SUPPORT=1)
target_link_libraries(gpufl PRIVATE httplib::httplib)


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

if(GPUFL_ENABLE_METAL)
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_METAL=1)
else()
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_METAL=0)
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
        # gpufl targets CUDA 13.x only; CUDA 11.x/12.x are not supported.
        if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_LESS 13)
            message(FATAL_ERROR
                "gpufl requires CUDA 13.x or newer; found ${CUDAToolkit_VERSION}.")
        endif()
        if(CUDAToolkit_FOUND)
            set(GPUFL_HAS_CUDA 1)
            target_sources(gpufl PRIVATE
                include/gpufl/backends/nvidia/sampler/cupti_sass.cpp
                include/gpufl/backends/nvidia/sampler/cupti_sass.hpp
                include/gpufl/backends/nvidia/cuda_collector.cpp
                include/gpufl/backends/nvidia/cupti_activity_callbacks.cpp
                include/gpufl/backends/nvidia/cupti_activity_state.cpp
                include/gpufl/backends/nvidia/cupti_utils.cpp
                include/gpufl/backends/nvidia/capture_capability_resolver.cpp
                include/gpufl/backends/nvidia/cupti_capture_capabilities.cpp
                include/gpufl/backends/nvidia/cupti_engine_selection.cpp
                include/gpufl/backends/nvidia/cupti_runtime_support.cpp
                include/gpufl/backends/nvidia/cuda_cleanup_handler.cpp
                include/gpufl/backends/nvidia/resource_handler.cpp
                include/gpufl/backends/nvidia/kernel_launch_handler.cpp
                include/gpufl/backends/nvidia/graph_structure_handler.cpp
                include/gpufl/backends/nvidia/mem_transfer_handler.cpp
                include/gpufl/backends/nvidia/synchronization_handler.cpp
                include/gpufl/backends/nvidia/monitor_adapter_nvidia.cpp
                include/gpufl/backends/nvidia/cupti_backend.cpp
                include/gpufl/backends/nvidia/engine/pc_sampling_engine.cpp
                include/gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.cpp
                include/gpufl/backends/nvidia/engine/pm_sampling_engine.cpp
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
            # NVTX (NVIDIA Tools Extension) - lets GFL_SCOPE emit ranges
            # visible to Nsight Systems and captured via CUPTI marker
            # activity, plus lets the PyTorch integration (gpufl.torch)
            # bridge dispatch events into the profiling pipeline.
            # --------------------------------------------------------
            # NVTX is header-only (nvtx3/) on CUDA 13. Full toolkits (e.g.
            # Windows) ship it under the toolkit include; the slim Linux
            # manylinux CUDA image does not, so vendor the header-only NVTX3
            # distribution as the fallback. NVTX3_INCLUDE_DIR is reused by the
            # gpufl_inject target below, whose hard #include needs the header.
            unset(NVTX3_INCLUDE_DIR CACHE)
            find_path(NVTX3_INCLUDE_DIR NAMES nvtx3/nvToolsExt.h
                HINTS "${CUDAToolkit_INCLUDE_DIRS}"
                      "${CUDAToolkit_ROOT}/include"
                      "$ENV{CUDA_PATH}/include"
                NO_DEFAULT_PATH
            )
            if(NVTX3_INCLUDE_DIR)
                message(STATUS "Found NVTX headers at ${NVTX3_INCLUDE_DIR}")
            else()
                include(FetchContent)
                FetchContent_Declare(nvtx
                    GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
                    GIT_TAG        v3.5.0
                    GIT_SHALLOW    TRUE)
                # Header-only: download without running NVTX's own CMake. CMP0169
                # (CMake>=3.30) deprecates bare FetchContent_Populate; OLD keeps
                # the download-only form usable.
                if(POLICY CMP0169)
                    cmake_policy(SET CMP0169 OLD)
                endif()
                FetchContent_GetProperties(nvtx)
                if(NOT nvtx_POPULATED)
                    FetchContent_Populate(nvtx)
                endif()
                set(NVTX3_INCLUDE_DIR "${nvtx_SOURCE_DIR}/c/include"
                    CACHE PATH "Vendored NVTX3 headers" FORCE)
                message(STATUS "Vendored NVTX3 headers (v3.5.0) at ${NVTX3_INCLUDE_DIR}")
            endif()
            target_include_directories(gpufl PRIVATE ${NVTX3_INCLUDE_DIR})
            target_compile_definitions(gpufl PRIVATE GPUFL_HAS_NVTX=1)

            # --------------------------------------------------------
            # NVPERF Support (for GFL_PERF_SCOPE hardware counters)
            # --------------------------------------------------------
            # Gated on the option, not just detection: find_library caches, so
            # there is no command-line way to force a miss.
            if(GPUFL_ENABLE_PERFWORKS)
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
                    message(WARNING "NVPERF not found - GFL_PERF_SCOPE disabled at runtime")
                endif()
            else()
                set(GPUFL_HAS_PERFWORKS 0)
                message(STATUS "PerfWorks disabled by GPUFL_ENABLE_PERFWORKS=OFF")
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
        # NVML stub library lives under the CUDA toolkit. We search both
        # the versioned path (cuda-13.1, cuda-12.x, ...) AND the unversioned
        # /usr/local/cuda symlink that the nvidia/cuda:*-devel base images
        # ship - earlier this only listed the versioned path, which silently
        # missed NVML when find_package(CUDAToolkit) failed to populate
        # CUDAToolkit_ROOT inside scikit-build-core's isolated build env.
        find_library(NVML_LIBRARY NAMES nvidia-ml
                HINTS
                "${CUDAToolkit_LIBRARY_ROOT}/stubs"
                "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib/stubs"
                "${CUDAToolkit_ROOT}/lib64/stubs"
                "$ENV{CUDA_PATH}/targets/x86_64-linux/lib/stubs"
                "/usr/local/cuda/targets/x86_64-linux/lib/stubs"
                "/usr/local/cuda/lib64/stubs"
                "/usr/lib/wsl/lib"  # Common location for WSL2 users
        )

        if(NVML_LIBRARY)
            set(GPUFL_HAS_NVML 1)
            target_link_libraries(gpufl PRIVATE ${NVML_LIBRARY})
            message(STATUS "Found NVML (Linux): ${NVML_LIBRARY}")
        else()
            # Fail loudly. A silent fallback ships a wheel with no GPU
            # telemetry - job_start emits "devices": [] at runtime and the
            # user has no idea NVML was missing at build time. Force the
            # caller to either fix the path (-DNVML_LIBRARY=...) or
            # explicitly opt out (-DGPUFL_ENABLE_NVIDIA=OFF). This trade-off
            # is intentional: a hard build break here is much cheaper to
            # diagnose than a silent runtime gap downstream.
            message(FATAL_ERROR
                "NVML not found! GPUFL_HAS_NVML would be 0 and the resulting "
                "library would emit no GPU telemetry at runtime.\n"
                "  Either: pass -DNVML_LIBRARY=/path/to/libnvidia-ml.so "
                "(typically under the CUDA toolkit 'stubs' folder),\n"
                "  Or:     pass -DGPUFL_ENABLE_NVIDIA=OFF to explicitly "
                "build without NVIDIA telemetry support.")
        endif()
    endif()

    if(WIN32)
        # dbghelp: stack traces. pdh: PDH GPU/CPU perf counters.
        # psapi: EnumProcessModules for runtime NVTX injection path discovery.
        target_link_libraries(gpufl PRIVATE dbghelp pdh psapi)
    endif()

    # NVAPI capability (Windows only): the FB (frame-buffer) domain of
    # NvAPI_GPU_GetDynamicPstatesInfoEx gives GPU memory-controller utilization
    # on WDDM (consumer/GeForce), where NVML's util rates read 0% and PDH has no
    # memory-utilization counter at all. The GPU domain also backstops gpu_util.
    # NVAPI is open source under the MIT license (github.com/NVIDIA/nvapi), so by
    # default we pull it via FetchContent (no binaries committed to this repo),
    # pinned to a commit. A local copy (-DNVAPI_ROOT=... or a third_party/nvapi
    # checkout) takes precedence for offline builds. See third_party/nvapi/README.md.
    if(WIN32 AND GPUFL_HAS_NVML)
        # 1) Prefer a local copy (offline / pinned snapshot) if present.
        find_path(NVAPI_INCLUDE_DIR NAMES nvapi.h
            HINTS "${NVAPI_ROOT}" "$ENV{NVAPI_ROOT}"
                  "${CMAKE_SOURCE_DIR}/third_party/nvapi")
        find_library(NVAPI_LIBRARY NAMES nvapi64 nvapi
            HINTS "${NVAPI_ROOT}" "$ENV{NVAPI_ROOT}"
                  "${CMAKE_SOURCE_DIR}/third_party/nvapi"
            PATH_SUFFIXES amd64 x64 lib/x64)

        # 2) Otherwise fetch the MIT-licensed open-source SDK (headers + the
        #    prebuilt amd64/nvapi64.lib stub) into build/_deps. Pinned by SHA -
        #    the repo publishes no tags.
        if(NOT (NVAPI_INCLUDE_DIR AND NVAPI_LIBRARY))
            include(FetchContent)
            FetchContent_Declare(nvapi
                GIT_REPOSITORY https://github.com/NVIDIA/nvapi.git
                GIT_TAG        9b181ea572f680327fe01a14a0f1f41c78034104)  # R595, no tags -> pin SHA
            # The repo ships no CMakeLists, so download only (no add_subdirectory).
            # CMP0169 (CMake>=3.30) deprecates bare FetchContent_Populate; OLD
            # keeps the download-only form usable.
            if(POLICY CMP0169)
                cmake_policy(SET CMP0169 OLD)
            endif()
            FetchContent_GetProperties(nvapi)
            if(NOT nvapi_POPULATED)
                FetchContent_Populate(nvapi)
            endif()
            find_path(NVAPI_INCLUDE_DIR NAMES nvapi.h HINTS "${nvapi_SOURCE_DIR}")
            find_library(NVAPI_LIBRARY NAMES nvapi64 nvapi
                HINTS "${nvapi_SOURCE_DIR}" PATH_SUFFIXES amd64 x64)
        endif()

        if(NVAPI_INCLUDE_DIR AND NVAPI_LIBRARY)
            set(GPUFL_HAS_NVAPI 1)
            target_include_directories(gpufl PRIVATE ${NVAPI_INCLUDE_DIR})
            target_link_libraries(gpufl PRIVATE ${NVAPI_LIBRARY})
            message(STATUS "Found NVAPI: ${NVAPI_LIBRARY} (mem_util/gpu_util on WDDM enabled)")
        else()
            message(STATUS
                "NVAPI unavailable - GPUFL_HAS_NVAPI=0. mem_util will read 0 on "
                "Windows WDDM (consumer GPUs). See third_party/nvapi/README.md.")
        endif()
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

    if(GPUFL_ENABLE_AMD_ROCPROFILER)
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
    else()
        message(STATUS "ROCprofiler-SDK support disabled by GPUFL_ENABLE_AMD_ROCPROFILER=OFF")
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

# -----------------------
# Metal backend (phase-1: buildable stub path)
# -----------------------
if(GPUFL_ENABLE_METAL)
    if(APPLE)
        enable_language(OBJCXX)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            set(GPUFL_HAS_METAL 1)
            target_sources(gpufl PRIVATE
                include/gpufl/backends/metal/metal_collector.mm
            )
            target_link_libraries(gpufl PRIVATE
                ${METAL_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
            )
            message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
        else()
            message(WARNING "GPUFL_ENABLE_METAL=ON but Metal/Foundation framework not found; GPUFL_HAS_METAL=0.")
        endif()
    else()
        message(WARNING "GPUFL_ENABLE_METAL=ON is only supported on macOS; GPUFL_HAS_METAL=0.")
    endif()
endif()

target_compile_definitions(gpufl PUBLIC
    GPUFL_HAS_CUDA=${GPUFL_HAS_CUDA}
    GPUFL_HAS_NVML=${GPUFL_HAS_NVML}
    GPUFL_HAS_NVAPI=${GPUFL_HAS_NVAPI}
    GPUFL_HAS_CUPTI=${GPUFL_HAS_CUPTI}
    GPUFL_HAS_PERFWORKS=${GPUFL_HAS_PERFWORKS}
    GPUFL_HAS_ROCM=${GPUFL_HAS_ROCM}
    GPUFL_HAS_ROCM_SMI=${GPUFL_HAS_ROCM_SMI}
    GPUFL_HAS_HIP=${GPUFL_HAS_HIP}
    GPUFL_HAS_ROCPROFILER_SDK=${GPUFL_HAS_ROCPROFILER_SDK}
    GPUFL_HAS_METAL=${GPUFL_HAS_METAL}
)

# -----------------------
# gpufl_inject - injection-mode shared library (Linux .so + Windows .dll).
#
# Loaded into a target process by `gpufl trace`: on Linux via LD_PRELOAD +
# CUDA_INJECTION64_PATH; on Windows via CUDA_INJECTION64_PATH only (no
# preload). The NVIDIA-defined InitializeInjection ABI (and, on Linux, an
# opt-in constructor) routes through std::call_once into gpufl::init() with
# injection_mode_default_options() (or the light/monitoring preset), with
# env-var overrides applied. See include/gpufl/inject/inject_entry.cpp.
# -----------------------
if(BUILD_GPUFL_INJECT AND ((UNIX AND NOT APPLE) OR WIN32))
    add_library(gpufl_inject SHARED
        include/gpufl/inject/inject_entry.cpp
    )
    if(UNIX AND NOT APPLE AND GPUFL_HAS_CUDA)
        # Compile CUDA boundary wrappers against the toolkit's official ABI.
        # inject_entry.cpp keeps its header-free fallback for no-CUDA builds.
        target_sources(gpufl_inject PRIVATE
            include/gpufl/inject/cuda_interpose_linux.cpp
        )
        target_compile_definitions(gpufl_inject PRIVATE
            GPUFL_TYPED_CUDA_INTERPOSE=1
        )
    endif()
    target_link_libraries(gpufl_inject PRIVATE gpufl::gpufl)

    # The injected DLL is loaded by the CUDA driver into the profiled target,
    # whose DLL search path has no reason to include our bin directory. The
    # provider resolves the runtime from the injection DLL's own location, so
    # it has to be there.
    add_custom_command(TARGET gpufl_inject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:gpufl_counter_runtime>
                $<TARGET_FILE_DIR:gpufl_inject>
        COMMENT "Colocating gpufl_counter_runtime with gpufl_inject"
    )
    if(CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(gpufl_inject PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(gpufl_inject PRIVATE ${NVTX3_INCLUDE_DIR})
    endif()
    set_target_properties(gpufl_inject PROPERTIES OUTPUT_NAME gpufl_inject)
    if(MSVC)
        # Emit a PDB for Release too - this DLL runs inside an injected, unattended
        # target process, so a crash there has no debugger attached; Windows Error
        # Reporting only gives a module+offset, which needs a matching PDB to
        # resolve to a function/line. /OPT:REF keeps dead-code elimination;
        # /OPT:NOICF disables identical-code folding, which otherwise merges
        # byte-identical functions and can attribute a crash to the wrong one.
        target_compile_options(gpufl_inject PRIVATE $<$<CONFIG:Release>:/Zi>)
        target_link_options(gpufl_inject PRIVATE
            $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:NOICF>)
    endif()
    if(UNIX AND NOT APPLE)
        set_target_properties(gpufl_inject PROPERTIES
            # Hide internal symbols; the extern "C" injection ABI and explicit
            # post-cuInit readiness handshake are marked visibility("default")
            # in the source.
            # The .init_array constructor is invoked by ld.so directly and
            # doesn't need to be a visible symbol.
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
        )
        install(TARGETS gpufl_inject
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    else()
        # Windows: exports come from __declspec(dllexport) in the source;
        # the DLL lives beside the launcher exe (RUNTIME), not in lib/.
        install(TARGETS gpufl_inject
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

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
                "${_DIR}/nvperf_target*.dll"
                # PC sampling loads pcsamplingutil by versionless name at
                # runtime; it must travel with our cupti64 so the in-process
                # PerfWorks preload can pin the MATCHING copy (a stale one on
                # the loader path would mismatch, like torch's nvperf_host).
                "${_DIR}/pcsamplingutil*.dll")
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
    # Keep googletest's install rules out of the package: the install-tree
    # consumer test installs this build into a scratch prefix, and gtest
    # headers and archives are not something gpufl_client ships.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(tests)
endif()

if(BUILD_GPUFL_MONITOR AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(daemon/monitor)
endif()

if(BUILD_GPUFL_LAUNCHER AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(daemon/launcher)
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

    # Same for the Python extension: it may be the first module to bind, in
    # which case it is the one that loads the runtime for the whole process.
    add_custom_command(TARGET _gpufl_client POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:gpufl_counter_runtime>
                $<TARGET_FILE_DIR:_gpufl_client>
        COMMENT "Colocating gpufl_counter_runtime with _gpufl_client"
    )

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
    COMPONENT counters
)

# Two export sets, deliberately. The generated targets file REFUSES to load
# when any imported target it references is absent, and the SDK records its
# private dependencies (httplib, zlib, OpenSSL, CUDA) in its link interface -
# so shipping counters in the SDK's set made `find_package(gpufl_client)` fail
# for a counters-only consumer with "httplib::httplib is missing". Measured,
# not theoretical. The counters set references nothing but system dl/pthread
# and loads anywhere; the SDK set loads only for consumers that opt in AND
# have its dependencies.
install(TARGETS gpufl_counters
    EXPORT gpufl_countersTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT counters
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT gpufl_countersTargets
    FILE gpufl_countersTargets.cmake
    NAMESPACE gpufl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpufl_client
    COMPONENT counters
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

# What makes `find_package(gpufl_client)` actually resolve: CMake looks for
# gpufl_clientConfig.cmake, and a bare Targets file is not one. Version file
# alongside so `find_package(gpufl_client 1.2)` can hold a floor.
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpufl_clientConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gpufl_clientConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpufl_client
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gpufl_clientConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gpufl_clientConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gpufl_clientConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpufl_client
    COMPONENT counters
)

# Cross-module counter sharing. Cannot be proven from inside one executable - it
# needs the Python extension AND the shared runtime as separate modules - so it
# runs as its own process rather than as a gtest case.
#
# Lives here, not in tests/, because tests/ is processed before _gpufl_client is
# defined; gating on that target there is always false and the check silently
# never registers - which is how it went unrun in the first place.
#
# Staged into the build tree rather than run against the source tree: the
# extension has to sit INSIDE the package to be importable, and copying a build
# artifact into source would leave it there for every later run.
if(BUILD_TESTING AND BUILD_PYTHON AND TARGET _gpufl_client
   AND TARGET gpufl_counter_runtime
   AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # The interpreter the EXTENSION was built for, not whichever one a fresh
    # find_package picks. A different minor version cannot import a cp3XX
    # module at all, so the check would silently run against the stub and
    # report that the registries are split when nothing of the sort happened.
    if(DEFINED Python_EXECUTABLE)
        set(GPUFL_XMOD_PYTHON "${Python_EXECUTABLE}")
    elseif(DEFINED PYTHON_EXECUTABLE)
        set(GPUFL_XMOD_PYTHON "${PYTHON_EXECUTABLE}")
    endif()
    if(GPUFL_XMOD_PYTHON)
        set(GPUFL_XMOD_STAGE ${CMAKE_BINARY_DIR}/xmod_stage)
        add_custom_target(counter_xmod_stage ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/python/gpufl
                    ${GPUFL_XMOD_STAGE}/python/gpufl
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:_gpufl_client>
                    ${GPUFL_XMOD_STAGE}/python/gpufl/
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:gpufl_counter_runtime>
                    ${GPUFL_XMOD_STAGE}/python/gpufl/
            COMMENT "Staging the cross-module counter check")
        add_dependencies(counter_xmod_stage _gpufl_client gpufl_counter_runtime)

        add_test(NAME counter_cross_module
                 COMMAND ${GPUFL_XMOD_PYTHON}
                         ${CMAKE_SOURCE_DIR}/scripts/counter_cross_module_check.py)
        set_tests_properties(counter_cross_module PROPERTIES
            ENVIRONMENT "GPUFL_REPO=${GPUFL_XMOD_STAGE}")
    endif()
endif()

# The install-tree consumer, as CTest rather than a manual run: install into a
# scratch prefix, configure the fixture consumer against it with find_package,
# build it, run it. This is the only test that can see EXPORT_NAME, the Config
# file install, and the counters export set's isolation from the SDK's
# dependencies - the unit suite links the build tree and passes no matter what
# the package ships.
if(BUILD_TESTING)
    enable_testing()
    add_test(NAME counters_package_consumer
             COMMAND ${CMAKE_COMMAND}
                     -DGPUFL_BINARY_DIR=${CMAKE_BINARY_DIR}
                     -DGPUFL_SOURCE_DIR=${CMAKE_SOURCE_DIR}
                     -DGPUFL_CONFIG=$<CONFIG>
                     -DGPUFL_GENERATOR=${CMAKE_GENERATOR}
                     -DGPUFL_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
                     -DGPUFL_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
                     -P ${CMAKE_SOURCE_DIR}/tests/package/counters_consumer_check.cmake)
    # Self-contained: the script builds gpufl_counters itself and installs
    # ONLY the counters component, so this test does not depend on - and
    # cannot be broken by - the rest of the project's install rules. A Debug
    # run without gpufl_counter_runtime built proved why that isolation
    # matters.
    set_tests_properties(counters_package_consumer PROPERTIES TIMEOUT 300)
endif()
