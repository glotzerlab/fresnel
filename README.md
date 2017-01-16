## C++ Documentation

Developers may find C++ level API documentation useful when modifying the code. To build it, simply execute
`doxygen` in the repository root. It will write HTML output in `devdoc/html/index.html`.

## Compiling

Compile and build with cmake.

```
mkdir build
cd build
cmake /path/to/fresnel
make -j4
```

## Prerequisites

* C++11 capable compiler
* Python >= 2.7
* For GPU raytracing (default off, requires `ENABLE_CUDA=ON` and `ENABLE_OPTIX=ON`):
    * OptiX >= 4.0
    * CUDA >= 7.5
* For CPU raytracing (default on, requires `ENABLE_TBB=ON` and `ENABLE_EMBREE=ON`):
    * Intel TBB >= 4.3.20150611
    * Intel Embree >= 2.10.0

Search paths:

| Library | Default search path | CMake Custom search path |
| ------- | ------------------- | ------------------ |
| OptiX   | `/opt/optix`        | `-DOptiX_INSTALL_DIR=/path/to/optix` |
| TBB     | *system*            | `TBB_LINK=/path/to/tbb/lib` (env var) |
| Embree  | *system*            | `EMBREE_LINK=/path/to/embree/lib` (env var) |
| Python  | $PATH               | `-DPYTHON_EXECUTABLE=/path/to/bin/python` |

On the first run of cmake, libraries that are found will automatically set the corresponding `ENABLE_library` **ON**.
Libraries are that not found will set ``ENABLE_library`` **OFF**. You can force off the use of a given library
on the cmake command line: e.g. `cmake -DENABLE_EMBREE=off`, or by changing these options in `ccmake`.
