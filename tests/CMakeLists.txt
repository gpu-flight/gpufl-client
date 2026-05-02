# tests/CMakeLists.txt

# 1. Bring in GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 2. Create the test executable
set(GPUFL_TEST_SOURCES
    main_test_runner.cpp
    core/test_analyzer.cpp
    core/test_batch_models.cpp
    core/test_monitor.cpp
    core/test_http_log_sink.cpp
    core/test_itanium_demangle.cpp
)

if(GPUFL_ENABLE_NVIDIA AND GPUFL_HAS_CUPTI)
  list(APPEND GPUFL_TEST_SOURCES
    backends/nvidia/test_nvidia_backend.cpp
    common/log_utils.cpp
  )
  # The engine-coverage test drives a real CUDA kernel; only include it when
  # CUDA language is available.
  if(GPUFL_HAS_CUDA)
    list(APPEND GPUFL_TEST_SOURCES
      backends/nvidia/test_engine_coverage.cpp
      common/test_kernel.cu
    )
  endif()
endif()

if(GPUFL_ENABLE_NVIDIA AND GPUFL_HAS_CUDA)
  list(APPEND GPUFL_TEST_SOURCES
    backends/nvidia/test_cuda_collector.cpp
  )
endif()

if(GPUFL_ENABLE_NVIDIA AND GPUFL_HAS_NVML)
  list(APPEND GPUFL_TEST_SOURCES
    backends/nvidia/test_nvml_collector.cpp
  )
endif()

if(GPUFL_ENABLE_AMD AND (GPUFL_HAS_ROCM_SMI OR GPUFL_HAS_HIP))
  list(APPEND GPUFL_TEST_SOURCES
    backends/amd/test_rocm_collector.cpp
  )
endif()

add_executable(gpufl_tests ${GPUFL_TEST_SOURCES})

# 3. Enable CUDA-specific test target behavior only when CUDA is available
if(GPUFL_HAS_CUDA)
  set_target_properties(gpufl_tests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# 4. CRITICAL: Point to your project's header structure
target_include_directories(gpufl_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include          # So we can do #include "gpufl/core/..."
    ${CMAKE_SOURCE_DIR}/tests            # So we can do #include "common/..."
)

# cpp-httplib's CPPHTTPLIB_ZLIB_SUPPORT propagates from gpufl
# (declared PUBLIC there), so the test target's httplib::Server
# handles incoming gzipped request bodies the same way the
# production backend does. No extra config needed here.

if(CUDAToolkit_INCLUDE_DIRS)
  target_include_directories(gpufl_tests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# CUPTI include directory is often not in CUDAToolkit_INCLUDE_DIRS
# In the root CMakeLists.txt it might be in CUDA::cupti target or CUPTI_INCLUDE_DIR variable.
if(TARGET CUDA::cupti)
    target_link_libraries(gpufl_tests PRIVATE CUDA::cupti)
endif()

if(CUPTI_INCLUDE_DIR)
    target_include_directories(gpufl_tests PRIVATE ${CUPTI_INCLUDE_DIR})
endif()

# 5. Link libraries
target_link_libraries(gpufl_tests PRIVATE
    gtest_main
    gpufl
)

# HttpLogSink tests spin up an embedded httplib::Server; link the test
# binary against the same httplib::httplib interface target the main
# library uses so headers and TLS config stay consistent.
if(TARGET httplib::httplib)
    target_link_libraries(gpufl_tests PRIVATE httplib::httplib)
endif()

if(TARGET CUDA::cudart)
    target_link_libraries(gpufl_tests PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

# 6. Copy CUPTI and CUDA DLLs on Windows for test execution
if(WIN32)
    set(CUPTI_DLL_FOUND FALSE)

    # Strategy 1: Try to find it via the CUDAToolkit_ROOT variable
    if(CUDAToolkit_ROOT)
        file(TO_CMAKE_PATH "${CUDAToolkit_ROOT}/extras/CUPTI/lib64" MANUAL_CUPTI_PATH)
        file(GLOB FOUND_DLLS "${MANUAL_CUPTI_PATH}/cupti64*.dll")
        if(FOUND_DLLS)
            set(CUPTI_DLL_FOUND TRUE)
            set(CUPTI_DLL_LIST ${FOUND_DLLS})
        endif()
    endif()

    # Strategy 2: Try the Target Property
    if(NOT CUPTI_DLL_FOUND AND TARGET CUDA::cupti)
        get_target_property(CUPTI_LIB_PATH CUDA::cupti IMPORTED_IMPLIB)
        if(NOT CUPTI_LIB_PATH)
             get_target_property(CUPTI_LIB_PATH CUDA::cupti IMPORTED_LOCATION)
        endif()
        if(CUPTI_LIB_PATH)
            get_filename_component(CUPTI_DIR "${CUPTI_LIB_PATH}" DIRECTORY)
            file(GLOB FOUND_DLLS "${CUPTI_DIR}/cupti64*.dll")
            if(FOUND_DLLS)
                set(CUPTI_DLL_FOUND TRUE)
                set(CUPTI_DLL_LIST ${FOUND_DLLS})
            endif()
        endif()
    endif()

    # Strategy 3: Fallback to Environment Variable
    if(NOT CUPTI_DLL_FOUND)
        file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/extras/CUPTI/lib64" ENV_CUPTI_PATH)
        file(GLOB FOUND_DLLS "${ENV_CUPTI_PATH}/cupti64*.dll")
        if(FOUND_DLLS)
            set(CUPTI_DLL_FOUND TRUE)
            set(CUPTI_DLL_LIST ${FOUND_DLLS})
        endif()
    endif()

    if(CUPTI_DLL_FOUND)
        list(GET CUPTI_DLL_LIST 0 CUPTI_DLL_TO_COPY)
        add_custom_command(TARGET gpufl_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUPTI_DLL_TO_COPY}"
            "$<TARGET_FILE_DIR:gpufl_tests>"
            COMMENT "Copying CUPTI DLL to test directory"
        )
    endif()

    # Also copy CUDA runtime DLLs needed for test discovery
    if(CUDAToolkit_ROOT)
        file(TO_CMAKE_PATH "${CUDAToolkit_ROOT}/bin" CUDA_BIN_PATH)
        file(GLOB CUDA_RUNTIME_DLLS
            "${CUDA_BIN_PATH}/cudart64*.dll"
            "${CUDA_BIN_PATH}/nvrtc64*.dll"
        )
        foreach(DLL_FILE ${CUDA_RUNTIME_DLLS})
            add_custom_command(TARGET gpufl_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DLL_FILE}"
                "$<TARGET_FILE_DIR:gpufl_tests>"
                COMMENT "Copying CUDA runtime DLL: ${DLL_FILE}"
            )
        endforeach()
    endif()
endif()

# 7. Register with CTest
include(GoogleTest)
# On Windows, skip automatic test discovery entirely and just run the test executable
# This avoids DLL dependency issues during the build phase
if(WIN32)
    add_test(NAME gpufl_tests COMMAND gpufl_tests)
    set_tests_properties(gpufl_tests PROPERTIES
        ENVIRONMENT "PATH=${CUDAToolkit_BIN_DIR};$ENV{PATH}"
    )
else()
    gtest_discover_tests(gpufl_tests
        DISCOVERY_TIMEOUT 20
        DISCOVERY_MODE POST_BUILD
    )
endif()
