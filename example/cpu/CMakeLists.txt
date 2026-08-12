add_executable(cpu_ray_tracing_circles ray_tracing_circles.cpp)
target_compile_features(cpu_ray_tracing_circles PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(cpu_ray_tracing_circles PRIVATE /W4 /permissive-)
else()
    target_compile_options(cpu_ray_tracing_circles PRIVATE -Wall -Wextra -Wpedantic)
endif()
