cmake_minimum_required(VERSION 3.31)

add_executable(gpufl-monitor
    main.cpp
    monitor_runner.cpp
)

target_link_libraries(gpufl-monitor PRIVATE gpufl::gpufl)
gpufl_check_private_openssl(gpufl-monitor)

if(GPUFL_HAS_CUDA)
    target_link_libraries(gpufl-monitor PRIVATE CUDA::cudart)
endif()

if(GPUFL_HAS_NVML)
    if(TARGET CUDA::nvml)
        target_link_libraries(gpufl-monitor PRIVATE CUDA::nvml)
    elseif(NVML_LIBRARY)
        target_link_libraries(gpufl-monitor PRIVATE ${NVML_LIBRARY})
    else()
        message(WARNING "GPUFL_HAS_NVML is true, but no NVML import target or library is available for gpufl-monitor")
    endif()
endif()

if(GPUFL_WHEEL_LAYOUT)
    # Beside the launcher in site-packages/gpufl/bin. A top-level bin/ in
    # site-packages would take the place of the scripts directory under
    # `pip install --target`, and pip would then skip the `gpufl` entry.
    install(TARGETS gpufl-monitor
        RUNTIME DESTINATION ${GPUFL_WHEEL_BINDIR}
    )
else()
    install(TARGETS gpufl-monitor
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Copy required DLLs on Windows
if(WIN32)
    # CUPTI DLL
    if(CUDAToolkit_ROOT)
        file(TO_CMAKE_PATH "${CUDAToolkit_ROOT}/extras/CUPTI/lib64" CUPTI_DLL_PATH)
        file(GLOB CUPTI_DLLS "${CUPTI_DLL_PATH}/cupti64*.dll")
        file(GLOB NVPERF_DLLS
            "${CUPTI_DLL_PATH}/nvperf_host*.dll"
            "${CUPTI_DLL_PATH}/nvperf_target*.dll")
        foreach(DLL ${CUPTI_DLLS} ${NVPERF_DLLS})
            add_custom_command(TARGET gpufl-monitor POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" "$<TARGET_FILE_DIR:gpufl-monitor>"
                COMMENT "Copying ${DLL}")
        endforeach()
    endif()

    # NVML DLL
    find_file(NVML_DLL NAMES nvml.dll
        PATHS "$ENV{ProgramFiles}/NVIDIA Corporation/NVSMI" "$ENV{SystemRoot}/System32")
    if(NVML_DLL)
        add_custom_command(TARGET gpufl-monitor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVML_DLL}" "$<TARGET_FILE_DIR:gpufl-monitor>"
            COMMENT "Copying nvml.dll")
    endif()
endif()

