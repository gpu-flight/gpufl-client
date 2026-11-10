cmake_minimum_required(VERSION 3.31)

# The `gpufl` launcher (trace / upload / version) builds on Linux and
# Windows. `trace` has a per-OS backend: POSIX fork/exec+LD_PRELOAD on
# Linux, CreateProcess+CUDA_INJECTION64_PATH on Windows. `upload` and
# `version` are portable. macOS has no CUDA injection path - skip it.
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
    bootstrap_status.cpp
    cli_help.cpp
    cli_parse.cpp
    cli_parse_internal.cpp
    cli_subcommand_options.cpp
    cli_trace_options.cpp
    info_command.cpp
    trace_command_common.cpp
    trace_run_plan.cpp
    deep_window_env.cpp
    segmentation_env.cpp
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

# trace_command_common.cpp inspects and rewrites .log.gz files when it needs
# to synthesize a missing shutdown marker after an injected trace exits.
if(TARGET ZLIB::ZLIB)
    target_link_libraries(gpufl_launcher PRIVATE ZLIB::ZLIB)
elseif(TARGET zlibstatic)
    target_link_libraries(gpufl_launcher PRIVATE zlibstatic)
    target_include_directories(gpufl_launcher PRIVATE
        ${zlib_SOURCE_DIR}
        ${zlib_BINARY_DIR}
    )
endif()

target_include_directories(gpufl_launcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../monitor
)

target_compile_features(gpufl_launcher PRIVATE cxx_std_17)

if(GPUFL_WHEEL_LAYOUT)
    install(TARGETS gpufl_launcher
        RUNTIME DESTINATION ${GPUFL_WHEEL_BINDIR}
    )
else()
    install(TARGETS gpufl_launcher
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# The `gpufl` a Linux wheel puts on PATH: pip installs this into the
# environment's scripts directory, and it execs the launcher installed above in
# site-packages (see wheel_entry_paths.hpp). It looks only for the launcher of
# the interpreter the wheel is built for, so it needs that interpreter's
# directory name under lib/: python3.12, or python3.13t when free-threaded.
if(GPUFL_WHEEL_LAYOUT)
    if(NOT Python_EXECUTABLE)
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import os, sysconfig; print(os.path.basename(os.path.dirname(sysconfig.get_path('platlib', 'posix_prefix', vars={'platbase': '/'}))))"
        OUTPUT_VARIABLE GPUFL_WHEEL_PYTHON_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE gpufl_wheel_python_dir_rc
    )
    if(NOT gpufl_wheel_python_dir_rc EQUAL 0
       OR NOT GPUFL_WHEEL_PYTHON_DIR MATCHES "^python[0-9]+\\.[0-9]+t?$")
        message(FATAL_ERROR
            "gpufl wheel: cannot work out ${Python_EXECUTABLE}'s lib/ "
            "directory (got '${GPUFL_WHEEL_PYTHON_DIR}')")
    endif()
    message(STATUS "gpufl wheel: the PATH entry runs "
                   "lib/${GPUFL_WHEEL_PYTHON_DIR}/.../gpufl/bin/gpufl")

    add_executable(gpufl_wheel_entry wheel_entry.cpp wheel_entry_paths.cpp)
    target_compile_definitions(gpufl_wheel_entry PRIVATE
        GPUFL_WHEEL_PYTHON_DIR="${GPUFL_WHEEL_PYTHON_DIR}"
    )
    target_compile_features(gpufl_wheel_entry PRIVATE cxx_std_17)
    # Named gpufl like the launcher, so it builds in a directory of its own.
    set_target_properties(gpufl_wheel_entry PROPERTIES
        OUTPUT_NAME gpufl
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/wheel_entry
    )
    install(TARGETS gpufl_wheel_entry
        RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR}
    )
endif()

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

# The counter runtime travels with the inject DLL. Once injected, gpufl_inject
# resolves it from its OWN directory - the target process's search path is not
# ours to rely on - so it has to be wherever the inject DLL ended up.
if(WIN32 AND TARGET gpufl_counter_runtime)
    add_dependencies(gpufl_launcher gpufl_counter_runtime)
    add_custom_command(TARGET gpufl_launcher POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:gpufl_counter_runtime>
            $<TARGET_FILE_DIR:gpufl_launcher>
        COMMENT "Colocating gpufl_counter_runtime.dll next to gpufl.exe")
endif()

# On Windows, copy CUPTI / NVPERF DLLs next to gpufl.exe (= beside the
# colocated gpufl_inject.dll) so the inject DLL's dependencies resolve when
# the driver loads it into the target. trace_command_win.cpp also prepends
# this directory to the child's PATH for robustness.
if(WIN32 AND COMMAND gpufl_copy_runtime_dlls)
    gpufl_copy_runtime_dlls(gpufl_launcher)
endif()
