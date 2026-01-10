cmake_minimum_required(VERSION 3.30)

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvml)
    message(STATUS "CUDA::nvml target not found automatically. Searching manually...")
    find_library(NVML_LIB
            NAMES nvml        # Windows uses 'nvml', Linux uses 'nvidia-ml'
            HINTS ${CUDAToolkit_LIBRARY_DIR}
    )
    find_path(NVML_INC
            NAMES nvml.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS}
    )

    if(NVML_LIB AND NVML_INC)
        add_library(CUDA::nvml UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvml PROPERTIES

                IMPORTED_LOCATION "${NVML_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${NVML_INC}"
        )
    else()
        message(FATAL_ERROR "Could not find NVML library. Please check your CUDA installation.")
    endif()
endif()

# Define Executables
add_executable(test_occupancy test_occupancy.cu)
add_executable(gfl_block_example block_style_example.cu)
add_executable(system_monitor system_monitor.cu)
add_executable(check_conflict check_conflict.cu)

target_link_libraries(test_occupancy PRIVATE
    gpufl::gpufl
    CUDA::cupti
    CUDA::cudart
)

target_link_libraries(gfl_block_example PRIVATE
    gpufl::gpufl
    CUDA::cupti
    CUDA::cudart
)

# system_monitor already had it correct
target_link_libraries(system_monitor PRIVATE
    gpufl::gpufl
    CUDA::cudart
    CUDA::nvml
    CUDA::cupti
)

target_link_libraries(check_conflict PRIVATE
    gpufl::gpufl
    CUDA::cudart
    CUDA::nvml
    CUDA::cupti
)

# Properties
set_target_properties(gfl_block_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(system_monitor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(test_occupancy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(check_conflict PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Platform Specifics
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(system_monitor PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(system_monitor PRIVATE winhttp ws2_32)

    # -------------------------------------------------------------------------
    # Robust CUPTI DLL Detection
    # -------------------------------------------------------------------------
    set(CUPTI_DLL_FOUND FALSE)

    # Strategy 1: Try to find it via the CUDAToolkit_ROOT variable (Most reliable)
    # This variable is set by find_package(CUDAToolkit)
    if(CUDAToolkit_ROOT)
        file(TO_CMAKE_PATH "${CUDAToolkit_ROOT}/extras/CUPTI/lib64" MANUAL_CUPTI_PATH)
        file(GLOB FOUND_DLLS "${MANUAL_CUPTI_PATH}/cupti64*.dll")
        if(FOUND_DLLS)
            set(CUPTI_DLL_FOUND TRUE)
            set(CUPTI_DLL_LIST ${FOUND_DLLS})
            message(STATUS "Found CUPTI DLL (via CUDAToolkit_ROOT): ${CUPTI_DLL_LIST}")
        endif()
    endif()

    # Strategy 2: Try the Target Property (if Strategy 1 failed)
    if(NOT CUPTI_DLL_FOUND AND TARGET CUDA::cupti)
        get_target_property(CUPTI_LIB_PATH CUDA::cupti IMPORTED_IMPLIB)
        if(CUPTI_LIB_PATH)
            get_filename_component(CUPTI_DIR "${CUPTI_LIB_PATH}" DIRECTORY)
            file(GLOB FOUND_DLLS "${CUPTI_DIR}/cupti64*.dll")
            if(FOUND_DLLS)
                set(CUPTI_DLL_FOUND TRUE)
                set(CUPTI_DLL_LIST ${FOUND_DLLS})
                message(STATUS "Found CUPTI DLL (via Target): ${CUPTI_DLL_LIST}")
            endif()
        endif()
    endif()

    # Strategy 3: Fallback to Environment Variable (Last Resort)
    if(NOT CUPTI_DLL_FOUND)
        file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/extras/CUPTI/lib64" ENV_CUPTI_PATH)
        file(GLOB FOUND_DLLS "${ENV_CUPTI_PATH}/cupti64*.dll")
        if(FOUND_DLLS)
            set(CUPTI_DLL_FOUND TRUE)
            set(CUPTI_DLL_LIST ${FOUND_DLLS})
            message(STATUS "Found CUPTI DLL (via ENV): ${CUPTI_DLL_LIST}")
        endif()
    endif()

    # -------------------------------------------------------------------------
    # Execute Copy
    # -------------------------------------------------------------------------
    if(CUPTI_DLL_FOUND)
        # Handle case where GLOB returns multiple versions; just take the first one
        list(GET CUPTI_DLL_LIST 0 CUPTI_DLL_TO_COPY)

        foreach(TARGET_NAME test_occupancy system_monitor gfl_block_example)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUPTI_DLL_TO_COPY}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                COMMENT "Copying ${CUPTI_DLL_TO_COPY} to output directory..."
            )
        endforeach()
    else()
        message(WARNING "Could NOT find cupti64*.dll automatically.\n"
            "Checked locations:\n"
            "1. ${CUDAToolkit_ROOT}/extras/CUPTI/lib64\n"
            "2. Target Property of CUDA::cupti\n"
            "3. $ENV{CUDA_PATH}/extras/CUPTI/lib64\n"
            "Please copy the DLL manually to the build folder.")
    endif()
elseif(UNIX)
    # Linux-specific settings
    target_link_libraries(system_monitor PRIVATE pthread curl)
endif()