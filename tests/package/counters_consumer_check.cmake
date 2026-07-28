# Steps the counters package through the exact flow a user follows:
#
#   cmake --build <build> --target gpufl_counters
#   cmake --install <build> --prefix <scratch> --component counters
#   cmake -S consumer -DCMAKE_PREFIX_PATH=<scratch>
#   cmake --build consumer
#   ./app                         -> "consumer ok valid=1"
#
# Run by CTest (see the add_test in the root CMakeLists), not by hand: the one
# manual run that validated this flow proved nothing about the NEXT change.
# Every step's failure names the step, so a regression reads as "install-tree
# consumer broke at configure" rather than a bare non-zero exit.
#
# The install is the COUNTERS COMPONENT, and the build is the counters target,
# so this test responds to exactly what the package promises - the archive,
# the public headers, the Config/Version files, the gpufl::counters export and
# its Threads/dl propagation - and to nothing else. A full-tree install tied
# it to zlib's install rules, httplib's export set and whether
# gpufl_counter_runtime happened to be built in this configuration; a Debug
# run failed on that last one without a single counters file being wrong.

foreach(var GPUFL_BINARY_DIR GPUFL_SOURCE_DIR GPUFL_CONFIG GPUFL_GENERATOR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "counters_consumer_check: ${var} not set")
    endif()
endforeach()

set(scratch "${GPUFL_BINARY_DIR}/package_check")
file(REMOVE_RECURSE "${scratch}")

# Single-config generators expand $<CONFIG> to an empty string, and
# `--config ""` is an error rather than a no-op. The flag exists only when
# there is a configuration to name.
set(config_args)
if(GPUFL_CONFIG)
    set(config_args --config "${GPUFL_CONFIG}")
endif()

function(run_step name)
    execute_process(COMMAND ${ARGN}
                    RESULT_VARIABLE rc
                    OUTPUT_VARIABLE out
                    ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR
            "install-tree consumer broke at ${name} (exit ${rc}):\n${out}\n${err}")
    endif()
    set(step_output "${out}" PARENT_SCOPE)
endfunction()

# Built here rather than assumed: CTest promises nothing about what ran
# before this test, and an install of a never-built target is a confusing
# missing-file error instead of a compile error.
run_step("counters build"
    ${CMAKE_COMMAND} --build "${GPUFL_BINARY_DIR}"
    --target gpufl_counters ${config_args})

run_step("install"
    ${CMAKE_COMMAND} --install "${GPUFL_BINARY_DIR}"
    --prefix "${scratch}/prefix" --component counters ${config_args})

# The parent's platform/toolset travel along when they exist: a VS parent
# configured for a specific -A/-T must not have its consumer silently probe a
# different one.
set(gen_args)
if(DEFINED GPUFL_PLATFORM AND GPUFL_PLATFORM)
    list(APPEND gen_args -A "${GPUFL_PLATFORM}")
endif()
if(DEFINED GPUFL_TOOLSET AND GPUFL_TOOLSET)
    list(APPEND gen_args -T "${GPUFL_TOOLSET}")
endif()

run_step("consumer configure"
    ${CMAKE_COMMAND} -G "${GPUFL_GENERATOR}" ${gen_args}
    -S "${GPUFL_SOURCE_DIR}/tests/package/consumer"
    -B "${scratch}/consumer"
    "-DCMAKE_PREFIX_PATH=${scratch}/prefix"
    "-DCMAKE_BUILD_TYPE=${GPUFL_CONFIG}")

run_step("consumer build"
    ${CMAKE_COMMAND} --build "${scratch}/consumer" ${config_args})

# Single-config generators put the binary at the top; multi-config nests it.
set(app "${scratch}/consumer/app")
foreach(candidate "${scratch}/consumer/app"
                  "${scratch}/consumer/app.exe"
                  "${scratch}/consumer/${GPUFL_CONFIG}/app.exe"
                  "${scratch}/consumer/${GPUFL_CONFIG}/app")
    if(EXISTS "${candidate}")
        set(app "${candidate}")
        break()
    endif()
endforeach()

run_step("consumer run" "${app}")
if(NOT step_output MATCHES "consumer ok valid=1")
    message(FATAL_ERROR
        "install-tree consumer ran but did not report a valid counter:\n"
        "${step_output}")
endif()

message(STATUS "install-tree counters consumer: ok")
