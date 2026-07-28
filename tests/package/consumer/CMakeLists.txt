# The documented consumer, verbatim. This project is configured against a
# scratch INSTALL tree by tests/package/counters_consumer_check.cmake - it is
# the fixture that keeps `find_package(gpufl_client)` + `gpufl::counters`
# honest. If EXPORT_NAME, the Config file install, or the counters export
# set's isolation from the SDK's dependencies regresses, this is the test
# that fails; the unit suite links the build tree and cannot see any of it.
cmake_minimum_required(VERSION 3.20)
project(gpufl_counters_consumer CXX)

find_package(gpufl_client 1.2 REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE gpufl::counters)
