cmake_minimum_required(VERSION 3.31)

# Linux-only (v1). Windows DLL injection is a Tier 2 follow-up.
if(NOT (UNIX AND NOT APPLE))
    return()
endif()

add_executable(gpufl
    main.cpp
    cli_parse.cpp
    trace_command.cpp
)

# Link against the static gpufl lib for kClientVersion / kWireVersion
# (via core/version.hpp) and the inject_entry.hpp env-var constants.
# Most of gpufl's object files are unreferenced from the launcher and
# get DCE'd at link time with --gc-sections.
target_link_libraries(gpufl PRIVATE gpufl::gpufl)

target_compile_features(gpufl PRIVATE cxx_std_17)

install(TARGETS gpufl
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
