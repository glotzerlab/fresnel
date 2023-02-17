.. Copyright (c) 2016-2023 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Building from source
====================

To build the **fresnel** Python package from source:

1. `Install prerequisites`_::

   $ <package-manager> install cmake git python numpy pybind11 qhull embree3
   $ <package-manager> install pillow pytest

2. `Obtain the source`_::

   $ git clone --recursive https://github.com/glotzerlab/fresnel

3. `Configure`_::

   $ cmake -B build/fresnel -S fresnel

4. `Build the package`_::

   $ cmake --build build/fresnel

5. `Install the package`_ (optional)::

   $ cmake --install build/fresnel

To build the documentation from source (optional):

1. `Install prerequisites`_::

   $ <package-manager> install sphinx sphinx_rtd_theme nbsphinx ipython

2. `Build the documentation`_::

   $ sphinx-build -b html fresnel/doc build/fresnel-documentation

The sections below provide details on each of these steps.

.. _Install prerequisites:

Install prerequisites
---------------------

**fresnel** requires a number of tools and libraries to build. The options ``ENABLE_EMBREE`` and
``ENABLE_OPTIX`` each require additional libraries when enabled.

.. note::

    This documentation is generic. Replace ``<package-manager>`` with your package or module
    manager. You may need to adjust package names and/or install additional packages, such as
    ``-dev`` packages that provide headers needed to build **fresnel**.

.. tip::

    Create a `virtual environment`_, one place where you can install dependencies and
    **fresnel**::

        $ python3 -m venv fresnel-venv

    You will need to activate your environment before configuring **fresnel**::

        $ source fresnel-venv/bin/activate

**General requirements:**

- C++14 capable compiler (tested with GCC 7-12, Clang 6-14, Visual Studio 2019-2022)
- CMake >= 3.8
- pybind11 >= 2.2
- Python >= 3.6
- numpy
- Qhull >= 2015.2
- For CPU execution (required when ``ENABLE_EMBREE=ON``):

  - Intel TBB >= 4.3.20150611
  - Intel Embree >= 3.0.0

- For GPU execution (required when ``ENABLE_OPTIX=ON``):

  - OptiX >= 6.0, < 7.0
  - CUDA >= 10

**Optional runtime dependencies:**

- pyside2

**To run tests:**

- pillow
- pytest

**To build the documentation:**

- sphinx
- sphinx_rtd_theme
- nbsphinx
- ipython

.. _virtual environment: https://docs.python.org/3/library/venv.html

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_::

   $ git clone --recursive https://github.com/glotzerlab/fresnel

Release tarballs are also available on the `downloads page`_.

.. seealso::

    See the `git book`_ to learn how to work with Git repositories.

.. warning::

    **fresnel** uses Git submodules. Clone with the ``--recursive`` to clone the submodules.

    Execute ``git submodule update --init`` to fetch the submodules each time you switch branches
    and the submodules show as modified.

.. _downloads page: https://glotzerlab.engin.umich.edu/Downloads/fresnel
.. _git book: https://git-scm.com/book
.. _Git: https://git-scm.com/

.. _Configure:

Configure
---------

Use CMake_ to configure a **fresnel** build in the given directory. Pass ``-D<option-name>=<value>``
to ``cmake`` to set options on the command line. When modifying code, you only need to repeat the
build step to update your build - it will automatically reconfigure as needed.

.. tip::

    Use Ninja_ to perform incremental builds in less time::

        $ cmake -B build/fresnel -S fresnel -GNinja

.. tip::

    Place your build directory in ``/tmp`` or ``/scratch`` for faster builds. CMake_ performs
    out-of-source builds, so the build directory can be anywhere on the filesystem.

.. tip::

    Pass the following options to CMake_ to optimize the build for your processor:
    ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native``

.. important::

    When using a virtual environment, activate the environment and set the cmake prefix path
    before running CMake_: ``$ export CMAKE_PREFIX_PATH=<path-to-environment>``

**fresnel**'s cmake configuration accepts a number of options.

Options that find libraries and executables only take effect on a clean invocation of CMake. To set
these options, first remove ``CMakeCache.txt`` from the build directory and then run ``cmake`` with
these options on the command line.

- ``PYTHON_EXECUTABLE`` - Specify which ``python`` to build against. Example: ``/usr/bin/python3``.

  - Default: ``python3.X`` detected on ``$PATH``.

- ``<package-name>_DIR`` - Specify the location of a package.

  - Default: Found on the `CMake`_ search path.

Other option changes take effect at any time:

- ``ENABLE_EMBREE`` - When enabled, build the CPU backend using Embree (default: ``on``).
- ``BUILD_OPTIX`` - When enabled, build the GPU backend using OpTiX (default: ``off``).
- ``CMAKE_BUILD_TYPE`` - Sets the build type (case sensitive) Options:

  - ``Debug`` - Compiles debug information into the library and executables. Enables asserts to
    check for programming mistakes. **fresnel** will run slow when compiled in ``Debug`` mode,
    but problems are easier to identify.
  - ``RelWithDebInfo`` - Compiles with optimizations and debug symbols.
  - ``Release`` - (default) All compiler optimizations are enabled and asserts are removed.
    Recommended for production builds.

- ``CMAKE_INSTALL_PREFIX`` - Directory to install fresnel. Defaults to the root path of the found
  Python executable.
- ``PYTHON_SITE_INSTALL_DIR`` - Directory to install ``fresnel`` to relative to
  ``CMAKE_INSTALL_PREFIX``. Defaults to the ``site-packages`` directory used by the found Python
  executable.

.. _CMake: https://cmake.org/
.. _Ninja: https://ninja-build.org/

.. _Build the package:

Build the package
-----------------

The command ``cmake --build build/fresnel`` will build the **fresnel** Python package in the given
build directory. After the build completes, the build directory will contain a functioning Python
package.

.. note::

    Pass ``--config <CONFIG>`` to build a specific configuration when using a multi-configuration
    generator such as Visual Studio::

        cmake --build build/fresnel --config Release

.. note::

    When using a multi-configuration generator, the Python package is built in
    ``build/fresnel/<CONFIG>``.

.. _Install the package:

Install the package
-------------------

The command ``cmake --install build/fresnel`` installs the given **fresnel** build to
``${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}``. CMake autodetects these paths, but you can
set them manually in CMake.

.. note::

    Pass ``--config <CONFIG>`` to install a specific configuration when using a multi-configuration
    generator such as Visual Studio.

.. _Build the documentation:

Build the documentation
-----------------------

Run `Sphinx`_ to build the documentation with the command
``sphinx-build -b html fresnel/sphinx-doc build/fresnel-documentation``. Open the file
:file:`build/fresnel-documentation/index.html` in your web browser to view the documentation.

.. tip::

    When iteratively modifying the documentation, the sphinx options ``-a -n -W -T --keep-going``
    are helpful to produce docs with consistent links in the side panel and to see more useful error
    messages::

        $ sphinx-build -a -n -W -T --keep-going -b html \
            fresnel/sphinx-doc build/fresnel-documentation

.. _Sphinx: https://www.sphinx-doc.org/
