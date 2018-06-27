# Fresnel

Fresnel is a python library that ray traces publication quality images in real time.

Fresnel is in an early stage of development. User documentation is not yet posted online, and the API is not yet stable.

## Install binaries

Official binaries of fresnel are available via [conda](http://conda.pydata.org/docs/) through
the [glotzer channel](https://anaconda.org/glotzer).
To install fresnel, first download and install
[miniconda](http://conda.pydata.org/miniconda.html) following [conda's instructions](http://conda.pydata.org/docs/install/quick.html).
Then add the `glotzer` channel and install fresnel:

```bash
$ conda config --add channels glotzer
$ conda install fresnel
```

## Get the source

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/Downloads/

```bash
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/fresnel/fresnel-v0.5.0.tar.gz
```

Or, clone using git:

```bash
$ git clone --recursive  https://bitbucket.org/glotzer/fresnel.git
```

Fresnel uses git submodules. Either clone with the ``--recursive`` option, or execute ``git submodule update --init``
to fetch the submodules.

## Compiling

Compile and build with cmake.

```bash
$ mkdir build
$ cd build
$ cmake /path/to/fresnel
$ make -j4
```

## Running tests

```bash
$ export PYTHONPATH=/path/to/build
$ cd /path/to/fresnel
$ cd test
$ pytest
```

## User documentation

[fresnel-examples](https://bitbucket.org/glotzer/fresnel-examples/overview) provides a tutorial introduction to fresnel
with jupyter notebooks. View [static fresnel-examples](http://nbviewer.jupyter.org/github/joaander/fresnel-examples/blob/master/index.ipynb) with nbviewer, or clone the examples and execute them locally:

```bash
$ export PYTHONPATH=/path/to/fresnel/build
$ git clone https://bitbucket.org/glotzer/fresnel-examples.git
$ cd fresnel-examples
$ jupyter notebook index.ipynb
```

Build the user reference documentation with sphinx:

```bash
$ cd /path/to/fresnel
$ cd doc
$ make html
$ open build/html/index.html
```

## Prerequisites

* C++11 capable compiler
* Python >= 2.7
* For GPU raytracing (requires `ENABLE_CUDA=ON` and `ENABLE_OPTIX=ON`):
    * OptiX >= 4.0
    * CUDA >= 7.5
* For CPU raytracing (requires `ENABLE_TBB=ON` and `ENABLE_EMBREE=ON`):
    * Intel TBB >= 4.3.20150611
    * Intel Embree >= 3.0.0
* To execute tests:
    * pytest
    * pillow

## Optional dependencies

* pillow
    * To display rendered output in Jupyter notebooks automatically
* sphinx
    * To build the user documentation
* doxygen
    * To build developer documentation

## Search paths

| Library | Default search path | CMake Custom search path |
| ------- | ------------------- | ------------------ |
| OptiX   | `/opt/optix`        | `-DOptiX_INSTALL_DIR=/path/to/optix` |
| TBB     | *system*            | `TBB_LINK=/path/to/tbb/lib` (env var) |
| Embree  | *system*            | `-Dembree_DIR=/path/to/embree-3.x.y` (the directory containing embree-config.cmake) |
| Python  | $PATH               | `-DPYTHON_EXECUTABLE=/path/to/bin/python` |

On the first run of cmake, libraries that are found will automatically set the corresponding `ENABLE_library` **ON**.
Libraries are that not found will set ``ENABLE_library`` **OFF**. You can force off the use of a given library
on the cmake command line: e.g. `cmake -DENABLE_EMBREE=off`, or by changing these options in `ccmake`.

## C++ Documentation

To build the developer documentation, execute
`doxygen` in the repository root. It will write HTML output in `devdoc/html/index.html`.

## Change log

See [ChangeLog.md](ChangeLog.md).
