cmake_minimum_required(VERSION 3.31)

# The `gpufl` launcher (trace / upload / version) builds on Linux and
# Windows. `trace` has a per-OS backend: POSIX fork/exec+LD_PRELOAD on
# Linux, CreateProcess+CUDA_INJECTION64_PATH on Windows. `upload` and
# `version` are portable. macOS has no CUDA injection path — skip it.
if(APPLE OR NOT (UNIX OR WIN32))
    return()
endif()

if(WIN32)
    set(GPUFL_LAUNCHER_TRACE_IMPL trace_command_win.cpp)
else()
    set(GPUFL_LAUNCHER_TRACE_IMPL trace_command.cpp)
endif()

# CMake target name is `gpufl_launcher` to avoid colliding with the
# `gpufl` static library target in the top-level CMakeLists. The
# produced binary is still named `gpufl` via OUTPUT_NAME so users run
# `gpufl trace -- <cmd>` (not `gpufl_launcher trace -- <cmd>`).
add_executable(gpufl_launcher
    main.cpp
    agent_launcher.cpp
    cli_parse.cpp
    ${GPUFL_LAUNCHER_TRACE_IMPL}
    monitor_command.cpp
    ../monitor/monitor_runner.cpp
    upload_command.cpp
)

set_target_properties(gpufl_launcher PROPERTIES OUTPUT_NAME gpufl)

# Link against the static gpufl lib for kClientVersion / kWireVersion
# (via core/version.hpp) and the inject_entry.hpp env-var constants.
# Most of gpufl's object files are unreferenced from the launcher and
# get DCE'd at link time with --gc-sections.
target_link_libraries(gpufl_launcher PRIVATE gpufl::gpufl)

target_include_directories(gpufl_launcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../monitor
)

target_compile_features(gpufl_launcher PRIVATE cxx_std_17)

install(TARGETS gpufl_launcher
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# On Windows, colocate gpufl_inject.dll next to gpufl.exe so the launcher's
# "colocated" discovery candidate resolves in the build tree and the DLL
# travels with the binary (Windows finds DLLs beside the exe). Linux resolves
# libgpufl_inject.so via the build-root candidate, so no copy is needed there.
if(WIN32 AND TARGET gpufl_inject)
    add_dependencies(gpufl_launcher gpufl_inject)
    add_custom_command(TARGET gpufl_launcher POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gpufl_inject>
            $<TARGET_FILE_DIR:gpufl_launcher>
        COMMENT "Colocating gpufl_inject.dll next to gpufl.exe")
endif()

# On Windows, copy CUPTI / NVPERF DLLs next to gpufl.exe (= beside the
# colocated gpufl_inject.dll) so the inject DLL's dependencies resolve when
# the driver loads it into the target. trace_command_win.cpp also prepends
# this directory to the child's PATH for robustness.
if(WIN32 AND COMMAND gpufl_copy_runtime_dlls)
    gpufl_copy_runtime_dlls(gpufl_launcher)
endif()
