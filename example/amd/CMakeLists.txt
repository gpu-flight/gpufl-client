cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(gpufl_amd_examples LANGUAGES CXX)

  set(GPUFL_ENABLE_AMD ON CACHE BOOL "Enable AMD backends (ROCm when available)" FORCE)
  set(GPUFL_ENABLE_NVIDIA OFF CACHE BOOL "Enable NVIDIA backends (CUDA + NVML when available)" FORCE)
  set(BUILD_GPUFL_EXAMPLE OFF CACHE BOOL "Build gpufl example application" FORCE)
  set(BUILD_PYTHON OFF CACHE BOOL "Build Python bindings" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree." FORCE)

  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. ${CMAKE_BINARY_DIR}/gpufl-root)
endif()

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCm path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCm installation directory")
    endif()
  endif()
endif()

if(NOT DEFINED HIP_PATH)
  if(DEFINED ENV{HIP_PATH})
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP installation")
  else()
    set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to HIP installation")
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
set(CMAKE_HIP_ARCHITECTURES OFF)

find_package(HIP REQUIRED)
find_package(hip QUIET CONFIG)

set(AMD_EXAMPLE_SOURCES
  check_device.cpp
  gpufl_scope_demo.cpp
  vector_add_benchmark.cpp
)

set_source_files_properties(
  ${AMD_EXAMPLE_SOURCES}
  PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1
)

hip_add_executable(amd_check_device check_device.cpp)
hip_add_executable(amd_gpufl_scope_demo gpufl_scope_demo.cpp)
hip_add_executable(amd_vector_add_benchmark vector_add_benchmark.cpp)

if(TARGET hip::host)
  target_link_libraries(amd_check_device PRIVATE hip::host)
  target_link_libraries(amd_gpufl_scope_demo PRIVATE hip::host)
  target_link_libraries(amd_vector_add_benchmark PRIVATE hip::host)
endif()

target_link_libraries(amd_check_device PRIVATE gpufl::gpufl)
target_link_libraries(amd_gpufl_scope_demo PRIVATE gpufl::gpufl)
target_link_libraries(amd_vector_add_benchmark PRIVATE gpufl::gpufl)
