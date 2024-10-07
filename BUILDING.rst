.. Copyright (c) 2016-2024 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Building from source
====================

To build **fresnel** from source:

1. `Install prerequisites`_:

   .. code-block:: bash

       micromamba install cmake embree git ninja numpy pybind11 python qhull

   Install additional packages needed to run the unit tests:

   .. code-block:: bash

       micromamba install pillow pytest

2. `Obtain the source`_:

   .. code-block:: bash

       git clone --recursive git@github.com:glotzerlab/fresnel.git

3. Change to the repository directory:

   .. code-block:: bash

       cd fresnel

4. `Configure`_:

   .. code-block:: bash

       cmake -B build -S . -GNinja

5. `Build the package`_:

   .. code-block:: bash

       cd build

   .. code-block:: bash

       ninja

6. `Run tests`_:

   .. code-block:: bash

       python3 -m pytest fresnel

6. `Install the package`_ (optional):

   .. code-block:: bash

       ninja install

To build the documentation from source (optional):

1. `Install prerequisites`_:

   .. code-block:: bash

       micromamba install furo nbsphinx ipython sphinx-copybutton

2. `Build the documentation`_:

   .. code-block:: bash

       sphinx-build -b html doc html

The sections below provide details on each of these steps.

.. _Install prerequisites:

Install prerequisites
---------------------

You will need to install a number of tools and libraries to build **fresnel**. The options
``ENABLE_EMBREE`` and ``ENABLE_OPTIX`` each require additional libraries when enabled.

**General requirements:**

- **C++17** capable compiler
- **CMake**
- **pybind11**
- **Python**
- **numpy**
- **Qhull**
- For CPU execution (required when ``ENABLE_EMBREE=ON``):

  - **Intel TBB**
  - **Intel Embree**

- For GPU execution (required when ``ENABLE_OPTIX=ON``):

  - **OptiX** >= 6.0, < 7.0
  - **CUDA**

**Optional runtime dependencies:**

- **pyside2**

**To run tests:**

- **pillow**
- **pytest**

**To build the documentation:**

- **sphinx**
- **sphinx_rtd_theme**
- **nbsphinx**
- **ipython**

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_:

.. code-block:: bash

    git clone --recursive git@github.com:glotzerlab/fresnel.git

Release tarballs are also available on the `GitHub release pages`_.

.. seealso::

    See the `git book`_ to learn how to work with Git repositories.

.. important::

    **fresnel** uses Git submodules. Clone with the ``--recursive`` to clone the submodules.

    Execute ``git submodule update --init`` to fetch the submodules each time you switch branches
    and the submodules show as modified.

.. _GitHub release pages: https://github.com/glotzerlab/fresnel/releases
.. _git book: https://git-scm.com/book
.. _Git: https://git-scm.com/

.. _Configure:

Configure
---------

Use CMake_ to configure the **fresnel** build directory:

.. code-block:: bash

    cd {{ path/to/fresnel/repository }}

.. code-block:: bash

    cmake -B build -S . -GNinja

Pass ``-D<option-name>=<value>`` to ``cmake`` to set options on the command line.

Options that find libraries and executables take effect only on a clean invocation of CMake. To set
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

.. tip::

    Pass the following options to CMake_ to optimize the build for your processor:
    ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native``

.. warning::

    When using a ``conda-forge`` environment for development, make sure that the environment does
    not contain ``clang``, ``gcc``, or any other compiler or linker. These interfere with the native
    compilers on your system and will result in compiler errors when building, linker errors when
    running, or segmentation faults.

.. _CMake: https://cmake.org/
.. _Ninja: https://ninja-build.org/

.. _Build the package:

Build the package
-----------------

After configuring, build **fresnel** with:

.. code-block:: bash

    cd build

.. code-block:: bash

    ninja

The ``build`` directory now contains a fully functional **fresnel** package.
Execute ``ninja`` again any time you modify the code, test scripts, or CMake scripts.

.. tip::

    ``ninja`` will automatically execute ``cmake`` as needed. You do **NOT** need to execute
    ``cmake`` yourself every time you build fresnel.

.. _Run tests:

Run tests
^^^^^^^^^

Use `pytest`_ to execute unit tests:

.. code-block:: bash

   python3 -m pytest fresnel

.. _pytest: https://docs.pytest.org/

.. _Install the package:

Install the package
-------------------

Execute:

.. code-block:: bash

    ninja install

to install **fresnel** into your Python environment.

.. warning::

    This will *overwrite* any **fresnel** that you may have installed by other means.

To use the compiled **fresnel** without modifying your environment, set ``PYTHONPATH``::

    export PYTHONPATH={{ path/to/fresnel/repository/build }}

.. _Build the documentation:

Build the documentation
-----------------------

Run `Sphinx`_ to build HTML documentation::

    sphinx-build -b html doc html

Open the file :file:`html/index.html` in your web browser to view the documentation.

.. tip::

    Add the sphinx options ``-a -n -W -T --keep-going`` to produce docs with consistent links in
    the side panel and provide more useful error messages.

.. _Sphinx: https://www.sphinx-doc.org/
