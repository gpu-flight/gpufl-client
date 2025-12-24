cmake_minimum_required(VERSION 3.31)

project(gpufl_client
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Header-only GPU monitoring client library"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------
# Options
# -----------------------
option(GPUFL_ENABLE_NVIDIA "Enable NVIDIA backends (CUDA + NVML when available)" ON)
option(GPUFL_ENABLE_AMD    "Enable AMD backends (ROCm when available)" OFF)

option(BUILD_GPUFL_EXAMPLE "Build gpufl example application" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)

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
    include/gpufl/core/logger.cpp
    include/gpufl/core/sampler.cpp
    include/gpufl/core/runtime.cpp
    include/gpufl/core/gpufl.cpp
    include/gpufl/core/common.cpp
    include/gpufl/core/debug_logger.cpp
)

set(GPUFL_HAS_CUDA 0)
set(GPUFL_HAS_NVML 0)
set(GPUFL_HAS_ROCM 0)

# -----------------------
# Backends
# -----------------------
if(GPUFL_ENABLE_NVIDIA)
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_NVIDIA=1)
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
                include/gpufl/cuda/monitor.cpp
                include/gpufl/cuda/cupti_backend.cpp)
            target_compile_definitions(gpufl PUBLIC GPUFL_HAS_CUDA=1)
            target_link_libraries(gpufl PRIVATE CUDA::cudart)

            # --------------------------------------------------------
            # CUPTI Support (Added)
            # --------------------------------------------------------
            if (TARGET CUDA::cupti)
                target_link_libraries(gpufl PRIVATE CUDA::cupti)
                target_compile_definitions(gpufl PUBLIC GPUFL_HAS_CUPTI=1)
                message(STATUS "Found CUPTI via CUDAToolkit target")
            else()
                # Fallback: Manual search if the target is missing
                find_library(CUPTI_LIBRARY NAMES cupti
                    HINTS "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
                    "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib"
                    "$ENV{CUDA_PATH}/extras/CUPTI/lib64"
                )
                find_path(CUPTI_INCLUDE_DIR NAMES cupti.h
                    HINTS "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include"
                    "$ENV{CUDA_PATH}/extras/CUPTI/include"
                )

                if(CUPTI_LIBRARY AND CUPTI_INCLUDE_DIR)
                    target_link_libraries(gpufl PRIVATE ${CUPTI_LIBRARY})
                    target_include_directories(gpufl PRIVATE ${CUPTI_INCLUDE_DIR})
                    target_compile_definitions(gpufl PUBLIC GPUFL_HAS_CUPTI=1)
                    message(STATUS "Found CUPTI manually: ${CUPTI_LIBRARY}")
                endif()
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
            target_compile_definitions(gpufl PUBLIC GPUFL_HAS_NVML=1)

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
        find_library(NVML_LIBRARY NAMES nvidia-ml)

        if(NVML_LIBRARY)
            set(GPUFL_HAS_NVML 1)
            target_compile_definitions(gpufl PUBLIC GPUFL_HAS_NVML=1)
            target_link_libraries(gpufl PRIVATE ${NVML_LIBRARY})
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
# AMD backends (placeholder, auto-detect later)
# -----------------------
if(GPUFL_ENABLE_AMD)
    target_compile_definitions(gpufl PUBLIC GPUFL_ENABLE_AMD=1)

    # Keep your current source wired the same way; later you can replace with real ROCm detection.
    set(GPUFL_HAS_ROCM 1)
    target_compile_definitions(gpufl PUBLIC GPUFL_HAS_ROCM=1)

    target_sources(gpufl PRIVATE
        include/gpufl/backends/amd/rocm_collector.cpp
    )
endif()

if(BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13
    )
    FetchContent_MakeAvailable(pybind11)

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
