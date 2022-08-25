![windmap](/screenshot.jpg?raw=true)

An animated 3D wind map using CUDA and OpenGL.

Based on the explanation in [this article](https://blog.mapbox.com/how-i-built-a-wind-map-with-webgl-b63022b5537f). The [wind data image](/wind.png?raw=true) was taken from that article.

## Building

You need Qt6, GLM and the NVIDIA CUDA development toolkit.

```
$ cmake -B build -S .
$ cmake --build build --parallel
```

## Troubleshooting

### CMake can't find Qt6

Add the path to `lib/cmake` under the Qt installation directory to the `CMAKE_PREFIX_PATH` environment variable.

### CMake complains about nvcc

On my system, nvcc was trying to use C++17. Try adding `-DCMAKE_CUDA_FLAGS="-Xcompiler=-std=c++14"` to the first cmake command.

### "System has an unsupported display driver / cuda driver combination"

On Ubuntu 22.04, apparently the `nvidia-cuda-toolkit` package isn't compatible with the drivers in `nvidia-driver-515`. Try to remove the nvidia packages and install the 510 driver.
