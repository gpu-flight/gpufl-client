cmake_minimum_required(VERSION 3.31)

add_executable(gpufl-monitor main.cpp)

target_link_libraries(gpufl-monitor PRIVATE gpufl::gpufl)

if(GPUFL_HAS_CUDA)
    target_link_libraries(gpufl-monitor PRIVATE CUDA::cudart)
endif()

if(GPUFL_HAS_NVML)
    target_link_libraries(gpufl-monitor PRIVATE CUDA::nvml)
endif()

install(TARGETS gpufl-monitor
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
