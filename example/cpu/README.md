# CPU ray-traced circles

`ray_tracing_circles.cpp` is a self-contained C++17 ray tracer. It casts one
ray per pixel (plus four anti-aliasing samples), intersects rays with spheres
and a floor plane, and casts shadow and reflection rays. The spheres project as
circles in the resulting image. It uses only CPU threads and the C++ standard
library—there is no CUDA, HIP, graphics API, or image library dependency.

Build it with the repository's CMake project:

```powershell
cmake -S . -B build -DBUILD_GPUFL_EXAMPLE=ON
cmake --build build --target cpu_ray_tracing_circles --config Release
```

Run it from the build output directory. The optional arguments are output file,
width, and height:

```powershell
.\Release\cpu_ray_tracing_circles.exe circles.ppm 1280 720
```

On single-configuration generators, omit `Release\`. The program writes a
binary PPM (`P6`) image, a simple lossless format that many image viewers and
editors can open. The default is `ray_traced_circles.ppm` at 960 × 540.

To compile the example without CMake, use any C++17 compiler with thread
support, for example:

```powershell
cl /std:c++17 /EHsc /O2 ray_tracing_circles.cpp
```
