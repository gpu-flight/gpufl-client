# tests/CMakeLists.txt

# 1. Bring in GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 2. Create the test executable
add_executable(gpufl_tests 
    main_test_runner.cpp
    core/test_analyzer.cpp
    core/test_monitor.cpp
    backends/nvidia/test_nvidia_backend.cpp
    backends/nvidia/test_cuda_collector.cpp
    backends/nvidia/test_nvml_collector.cpp
)

# 3. Enable CUDA language for this target (assuming CMake 3.18+)
set_target_properties(gpufl_tests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gpufl_tests PROPERTIES LANGUAGE CUDA)

# 4. CRITICAL: Point to your project's header structure
target_include_directories(gpufl_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/include          # So we can do #include "gpufl/core/..."
    ${CMAKE_SOURCE_DIR}/tests            # So we can do #include "common/..."
    ${CUDAToolkit_INCLUDE_DIRS}
)

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
    )
endif()
