Installation
============

**fresnel** binaries are available in the glotzerlab-software_ Docker_/Singularity_ images and in
packages on conda-forge_. You can also compile **fresnel** from source

.. _glotzerlab-software: https://glotzerlab-software.readthedocs.io
.. _Docker: https://hub.docker.com/
.. _Singularity: https://www.sylabs.io/
.. _conda-forge: https://conda-forge.org/

Binaries
--------

Conda package
^^^^^^^^^^^^^

**fresnel** is available on conda-forge_ on the *linux-64*, *osx-64*, and *osx-arm64* platforms. To
install, download and install miniforge_ or miniconda_ Then install **fresnel** from the
conda-forge_ channel:

.. _miniforge: https://github.com/conda-forge/miniforge
.. _miniconda: http://conda.pydata.org/miniconda.html

.. code-block:: bash

   $ conda install -c conda-forge fresnel

.. note::

    The **fresnel** package on ``conda-forge`` does not support GPUs

Singularity / Docker images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the glotzerlab-software_ documentation for instructions to install and use the containers.

Compile from source
-------------------

Obtain the source
^^^^^^^^^^^^^^^^^

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/downloads/fresnel::

   $ curl -O https://glotzerlab.engin.umich.edu/downloads/fresnel/fresnel-v0.13.1.tar.gz

Or, clone using git::

   $ git clone --recursive  https://github.com/glotzerlab/fresnel

**Fresnel** uses git submodules. Either clone with the ``--recursive`` option, or execute ``git
submodule update --init`` to fetch the submodules.

Configure a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a shared Python installation, create a `virtual environment
<https://docs.python.org/3/library/venv.html>`_ where you can install
**fresnel**::

    $ python3 -m venv /path/to/virtual/environment

Activate the environment before configuring and before executing
**fresnel** scripts::

    $ source /path/to/virtual/environment/bin/activate

Tell CMake to search in the virtual environment first::

    $ export CMAKE_PREFIX_PATH=/path/to/virtual/environment

.. note::

   Other types of virtual environments (such as *conda*) may work, but are not thoroughly tested.

Install Prerequisites
^^^^^^^^^^^^^^^^^^^^^

**fresnel** requires:

* C++14 capable compiler
* CMake >= 3.8
* pybind11 >= 2.2
* Python >= 3.6
* numpy
* Qhull >= 2015.2
* For CPU execution (required when ``ENABLE_EMBREE=ON``):

  * Intel TBB >= 4.3.20150611
  * Intel Embree >= 3.0.0

* For GPU execution (required when ``ENABLE_OPTIX=ON``):

  * OptiX == 6.0
  * CUDA >= 10

``ENABLE_EMBREE`` (*defaults ON*) and ``ENABLE_OPTIX`` (*defaults OFF*) are orthogonal settings,
either or both may be enabled.

Additional packages may be needed:

* pyside2

    * Required t.o enable interactive widgets. (runtime)

* pillow

  * Required to display rendered output in Jupyter notebooks automatically. (runtime)
  * Required to execute unit tests.

* pytest

  * Required to execute unit tests.

* sphinx, sphinx_rtd_theme, and nbspinx

  * Required to build the user documentation.

* doxygen

  * Requited to build developer documentation.

Install these tools with your system or virtual environment package manager. **fresnel** developers
have had success with ``pacman`` (`arch linux <https://www.archlinux.org/>`_), ``apt-get`` (`ubuntu
<https://ubuntu.com/>`_), `Homebrew <https://brew.sh/>`_ (macOS), and `MacPorts
<https://www.macports.org/>`_ (macOS)::

    $ your-package-manager install cmake doxygen embree pybind11 python python-pillow python-pytest python-sphinx python-sphinx_rtd_theme python-nbsphinx intell-tbb qhull

Typical HPC cluster environments provide python, numpy, and cmake via a module system::

    $ module load gcc python cmake

.. note::

    Packages may be named differently, check your system's package list. Install any ``-dev``
    packages as needed.

.. tip::

    You can install numpy and other python packages into your virtual environment::

        python3 -m pip install numpy

Compile
^^^^^^^

Configure with **cmake** and compile with **make**::

   $ cd /path/to/fresnel
   $ mkdir build
   $ cd build
   $ cmake ../
   $ make install -j10

By default, **fresnel** builds the Embree (CPU) backend. Pass ``-DENABLE_OPTIX=ON`` to **cmake** to
enable the GPU accelerated OptiX backend.

Run tests
^^^^^^^^^

To run tests, execute ``pytest`` in the build directory or in an environment
where fresnel is installed to run all tests.

.. code-block:: bash

   $ pytest --pyargs fresnel

Build user documentation
^^^^^^^^^^^^^^^^^^^^^^^^

Build the user documentation with **sphinx**::

   $ cd /path/to/fresnel
   $ cd doc
   $ make html
   $ open build/html/index.html

Build C++ Documentation
^^^^^^^^^^^^^^^^^^^^^^^

To build the developer documentation, execute ``doxygen`` in the repository root. It will write HTML
output in ``devdoc/html/index.html``.
