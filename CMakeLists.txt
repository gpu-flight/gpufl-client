cmake_minimum_required(VERSION 3.18)

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

target_sources(gpufl PRIVATE
    include/gpufl/core/logger.cpp
    include/gpufl/core/sampler.cpp
    include/gpufl/core/runtime.cpp
    include/gpufl/core/gpufl.cpp
    include/gpufl/core/common.cpp
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
                include/gpufl/cuda/cuda.cpp)
            target_compile_definitions(gpufl PUBLIC GPUFL_HAS_CUDA=1)
            target_link_libraries(gpufl PRIVATE CUDA::cudart)
        endif()
    endif()
    #
    # NVML capability:
    # - On Linux, link libnvidia-ml if present.
    # - On Windows, many projects LoadLibrary/GetProcAddress at runtime.
    #   If you do runtime loading, treat NVML as "capable" without link-time lib.
    #

    if(WIN32)
        find_library(NVML_LIBRARY NAMES nvml
            PATHS
            "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
            "$ENV{CUDA_PATH}/lib/x64"
            PATH_SUFFIXES lib lib/x64
        )
    else()
        find_library(NVML_LIBRARY NAMES nvidia-ml)
    endif()

    if(NVML_LIBRARY)
        set(GPUFL_HAS_NVML 1)
        target_compile_definitions(gpufl PUBLIC GPUFL_HAS_NVML=1)
        target_link_libraries(gpufl PRIVATE ${NVML_LIBRARY})
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
