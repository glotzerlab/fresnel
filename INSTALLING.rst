Installation
============

**Fresnel** binaries are available in the `glotzerlab-software <https://glotzerlab-software.readthedocs.io>`_
`Docker <https://hub.docker.com/>`_/`Singularity <https://www.sylabs.io/>`_ images and in packages on
`conda-forge <https://conda-forge.org/>`_. You can also compile **fresnel** from source.

Binaries
--------

Anaconda package
^^^^^^^^^^^^^^^^

**Fresnel** is available on `conda-forge <https://conda-forge.org/>`_. To install, first download and install
`miniconda <http://conda.pydata.org/miniconda.html>`_.
Then add the ``conda-forge`` channel and install **fresnel**::

   ▶ conda config --add channels conda-forge
   ▶ conda install fresnel

**jupyter** and **matplotlib** are required to execute the
`fresnel example notebooks <https://github.com/glotzerlab/fresnel-examples>`_::

   ▶ conda install jupyter matplotlib

.. note::

    The **fresnel** package on ``conda-forge`` does not support GPUs

Singularity / Docker images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the `glotzerlab-software documentation <https://glotzerlab-software.readthedocs.io/>`_ for container usage
information and cluster specific instructions.

Compile from source
-------------------

Obtain the source
^^^^^^^^^^^^^^^^^

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/downloads/fresnel::

   ▶ curl -O https://glotzerlab.engin.umich.edu/downloads/fresnel/fresnel-v0.9.0.tar.gz

Or, clone using git::

   ▶ git clone --recursive  https://github.com/glotzerlab/fresnel

**Fresnel** uses git submodules. Either clone with the ``--recursive`` option, or execute ``git submodule update --init``
to fetch the submodules.

Configure a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a shared Python installation, create a `virtual environment
<https://docs.python.org/3/library/venv.html>`_ where you can install
**fresnel**::

    ▶ python3 -m venv /path/to/virtual/environment --system-site-packages

Activate the environment before configuring and before executing
**fresnel** scripts::

    ▶ source /path/to/virtual/environment/bin/activate

Tell CMake to search in the virtual environment first::

    ▶ export CMAKE_PREFIX_PATH=/path/to/virtual/environment

.. note::

   Other types of virtual environments (such as *conda*) may work, but are not thoroughly tested.

Install Prerequisites
^^^^^^^^^^^^^^^^^^^^^

**fresnel** requires:

* C++11 capable compiler
* CMake >= 3.8
* pybind11 >= 2.2
* Python >= 3.5
* numpy
* Qhull >= 2015.2
* For CPU execution (required when ``ENABLE_EMBREE=ON``):

  * Intel TBB >= 4.3.20150611
  * Intel Embree >= 3.0.0

* For GPU execution (required when ``ENABLE_OPTIX=ON``):

  * OptiX >= 4.0
  * CUDA >= 7.5

``ENABLE_EMBREE`` (*defaults ON*) and ``ENABLE_OPTIX`` (*defaults OFF*) are orthogonal settings, either or both may be
enabled.

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

Install these tools with your system or virtual environment package manager. **fresnel** developers have had success with
``pacman`` (`arch linux <https://www.archlinux.org/>`_), ``apt-get`` (`ubuntu <https://ubuntu.com/>`_), `Homebrew
<https://brew.sh/>`_ (macOS), and `MacPorts <https://www.macports.org/>`_ (macOS)::

    ▶ your-package-manager install cmake doxygen embree pybind11 python python-pillow python-pytest python-sphinx python-sphinx_rtd_theme python-nbsphinx intell-tbb qhull

Typical HPC cluster environments provide python, numpy, and cmake via a module system::

    ▶ module load gcc python cmake

.. note::

    Packages may be named differently, check your system's package list. Install any ``-dev`` packages as needed.

.. tip::

    You can install numpy and other python packages into your virtual environment::

        python3 -m pip install numpy

Compile
^^^^^^^

Configure with **cmake** and compile with **make**::

   ▶ cd /path/to/fresnel
   ▶ mkdir build
   ▶ cd build
   ▶ cmake ../
   ▶ make install -j10

By default, **fresnel** builds the Embree (CPU) backend. Pass ``-DENABLE_OPTIX=ON`` to **cmake** to enable the GPU
accelerated OptiX backend.

Run tests
^^^^^^^^^

To test **fresnel** builds without installing, add the build directory to your ``PYTHONPATH``::

   ▶ export PYTHONPATH=$PYTHONPATH:/path/to/fresnel/build

Execute ``pytest`` in the ``test`` directory to run all tests.

.. code-block:: bash

   ▶ cd /path/to/fresnel
   ▶ cd test
   ▶ pytest

Build user documentation
^^^^^^^^^^^^^^^^^^^^^^^^

Build the user documentation with **sphinx**::

   ▶ cd /path/to/fresnel
   ▶ cd doc
   ▶ make html
   ▶ open build/html/index.html

Build C++ Documentation
^^^^^^^^^^^^^^^^^^^^^^^

To build the developer documentation, execute
``doxygen`` in the repository root. It will write HTML output in ``devdoc/html/index.html``.
