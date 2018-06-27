# Fresnel

Fresnel is a python library that ray traces publication quality images in real time.

Fresnel is in an early stage of development. User documentation is not yet posted online, and the API is not yet stable.

## Installing Fresnel

Fresnel binary packages are available via [conda-forge](https://conda-forge.org/) and images via the
[Docker Hub](https://hub.docker.com/). You can also compile fresnel from source.

### Docker images

Pull the [glotzerlab/software](https://hub.docker.com/r/glotzerlab/software/) to get fresnel along with
many other tools commonly used in simulation/analysis workflows. Use these images to execute fresnel in
Docker/Singularity containers on Mac, Linux, and cloud systems you control and on HPC clusters with Singularity
support.

See full usage information on the [glotzerlab/software docker hub page](https://hub.docker.com/r/glotzerlab/software/).

+Singularity:

```bash
$ umask 002
$ singularity pull docker://glotzerlab/software
```

Docker:

```bash
$ docker pull glotzerlab/software
```

## Anaconda package

Fresnel is available on [conda-forge](https://conda-forge.org/). To install, first download and install
[miniconda](http://conda.pydata.org/miniconda.html) following [conda's instructions](http://conda.pydata.org/docs/install/quick.html).
Then add the `conda-forge` channel and install fresnel:

```bash
$ conda config --add channels conda-forge
$ conda install fresnel
```

## Compile from source

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

By default, fresnel builds the Embree (CPU) backend. Enable the GPU accelerated OptiX backend by passing
``-DENABLE_OPTIX=ON`` to cmake.

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
* For CPU execution (required when `ENABLE_EMBREE=ON`):
    * Intel TBB >= 4.3.20150611
    * Intel Embree >= 3.0.0
* For GPU execution (required when `ENABLE_OPTIX=ON`):
    * OptiX >= 4.0
    * CUDA >= 7.5
* To execute tests:
    * pytest
    * pillow

``ENABLE_EMBREE`` and ``ENABLE_OPTIX`` are orthogonal settings, either or both may be enabled.

## Optional dependencies

* pillow
    * To display rendered output in Jupyter notebooks automatically
* sphinx
    * To build the user documentation
* doxygen
    * To build developer documentation

## Search paths

OptiX, TBB, Embree, and Python may be installed in a variety of locations. Use these methods to specify
a specific library for fresnel to use the *first* time you invoke ``cmake`` in a clean build directory.

| Library | Default search path | CMake Custom search path |
| ------- | ------------------- | ------------------ |
| OptiX   | `/opt/optix`        | `-DOptiX_INSTALL_DIR=/path/to/optix` |
| TBB     | *system*            | `TBB_LINK=/path/to/tbb/lib` (env var) |
| Embree  | *system*            | `-Dembree_DIR=/path/to/embree-3.x.y` (the directory containing embree-config.cmake) |
| Python  | $PATH               | `-DPYTHON_EXECUTABLE=/path/to/bin/python` |

## C++ Documentation

To build the developer documentation, execute
`doxygen` in the repository root. It will write HTML output in `devdoc/html/index.html`.

## Change log

See [ChangeLog.md](ChangeLog.md).
