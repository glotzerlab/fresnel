## C++ Documentation

Developers may find C++ level API documentation useful when modifying the code. To build it, simply execute
`doxygen` in the repository root. It will write HTML output in `devdoc/html/index.html`.

## Compiling

Compile and build with cmake.

```mkdir build
cd build
cmake /path/to/fresnel
make -j20
```

## Prerequisites

* C++11 capable compiler
* Python >= 2.7
* For GPU raytracing (default off, requires `-DENABLE_CUDA=ON -DENABLE_OPTIX=ON`):
    * OptiX >= 4.0
    * CUDA >= 7.5
* For CPU raytracing (default on, requires `-DENABLE_TBB=ON -DENABLE_EMBREE=ON`):
    * Intel TBB >= 4.3.20150611
    * Intel Embree >= 2.10.0

Search paths:

| Library | Default search path | CMake Custom search path |
| ------- | ------------------- | ------------------ |
| OptiX   | `/opt/optix`        | `-DOptiX_INSTALL_DIR=/path/to/optix` |
| TBB     | *system*            | `-DTBB_LINK=/path/to/tbb/lib` |
| Embree  | *system*            | `-DEMBREE_LINK=/path/to/embree/lib` |
| Python  | $PATH               | `-DPYTHON_EXECUTABLE=/path/to/bin/python` |
