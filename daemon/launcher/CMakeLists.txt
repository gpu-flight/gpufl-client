cmake_minimum_required(VERSION 3.31)

# Linux-only (v1). Windows DLL injection is a Tier 2 follow-up.
if(NOT (UNIX AND NOT APPLE))
    return()
endif()

# CMake target name is `gpufl_launcher` to avoid colliding with the
# `gpufl` static library target in the top-level CMakeLists. The
# produced binary is still named `gpufl` via OUTPUT_NAME so users run
# `gpufl trace -- <cmd>` (not `gpufl_launcher trace -- <cmd>`).
add_executable(gpufl_launcher
    main.cpp
    cli_parse.cpp
    trace_command.cpp
)

set_target_properties(gpufl_launcher PROPERTIES OUTPUT_NAME gpufl)

# Link against the static gpufl lib for kClientVersion / kWireVersion
# (via core/version.hpp) and the inject_entry.hpp env-var constants.
# Most of gpufl's object files are unreferenced from the launcher and
# get DCE'd at link time with --gc-sections.
target_link_libraries(gpufl_launcher PRIVATE gpufl::gpufl)

target_compile_features(gpufl_launcher PRIVATE cxx_std_17)

install(TARGETS gpufl_launcher
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
