Installation
============

**Fresnel** binaries are available in the `glotzerlab-software <https://glotzerlab-software.readthedocs.io>`_
`Docker <https://hub.docker.com/>`_/`Singularity <https://www.sylabs.io/>`_ images and in packages on
`conda-forge <https://conda-forge.org/>`_. You can also compile **fresnel** from source.

Anaconda package
----------------

**Fresnel** is available on `conda-forge <https://conda-forge.org/>`_. To install, first download and install
`miniconda <http://conda.pydata.org/miniconda.html>`_.
Then add the ``conda-forge`` channel and install **fresnel**:

.. code-block:: bash

   $ conda config --add channels conda-forge
   $ conda install fresnel

**jupyter** and **matplotlib** are required to execute the
`fresnel example notebooks <https://github.com/glotzerlab/fresnel-examples>`_, install

.. code-block:: bash

   $ conda install jupyter matplotlib

You can update **fresnel** with:

.. code-block:: bash

   $ conda update fresnel

.. note::

    The **fresnel** package on ``conda-forge`` does not include GPU support.

Docker images
-------------

Pull the `glotzerlab-software <https://glotzerlab-software.readthedocs.io>`_ image to get
**fresnel** along with many other tools commonly used in simulation and analysis workflows. See full usage information in the
`glotzerlab-software documentation <https://glotzerlab-software.readthedocs.io>`_.

Singularity:

.. code-block:: bash

   $ singularity pull shub://glotzerlab/software

Docker:

.. code-block:: bash

   $ docker pull glotzerlab/software


Compile from source
-------------------

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/Downloads/fresnel

.. code-block:: bash

   $ curl -O https://glotzerlab.engin.umich.edu/Downloads/fresnel/fresnel-v0.7.1.tar.gz

Or, clone using git:

.. code-block:: bash

   $ git clone --recursive  https://github.com/glotzerlab/fresnel

**Fresnel** uses git submodules. Either clone with the ``--recursive`` option, or execute ``git submodule update --init``
to fetch the submodules.

Prerequisites
^^^^^^^^^^^^^

* C++11 capable compiler
* CMake >= 2.8
* Python >= 2.7
* For CPU execution (required when ``ENABLE_EMBREE=ON``):

  * Intel TBB >= 4.3.20150611
  * Intel Embree >= 3.0.0

* For GPU execution (required when ``ENABLE_OPTIX=ON``):

  * OptiX >= 4.0
  * CUDA >= 7.5

* To enable interactive widgets:

    * pyside2

* To execute tests (optional):

  * pytest
  * pillow

``ENABLE_EMBREE`` (*defaults ON*) and ``ENABLE_OPTIX`` (*defaults OFF*) are orthogonal settings, either or both may be
enabled.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

* pytest

  * Required to execute unit tests.

* pillow

  * Required to display rendered output in Jupyter notebooks automatically.

* sphinx

  * Required to build the user documentation.

* doxygen

  * Requited to build developer documentation.

Compile
^^^^^^^

Configure with **cmake** and compile with **make**. Replace ``${PREFIX}`` your desired installation location.

.. code-block:: bash

   $ mkdir build
   $ cd build
   $ cmake ../ -DCMAKE_INSTALL_PREFIX=${PREFIX}/lib/python
   $ make install -j10

By default, **fresnel** builds the Embree (CPU) backend. Pass ``-DENABLE_OPTIX=ON`` to **cmake** to enable the GPU
accelerated OptiX backend.

Add ``${PREFIX}/lib/python`` to your ``PYTHONPATH`` to use **fresnel**.

.. code-block:: bash

   $ export PYTHONPATH=$PYTHONPATH:${PREFIX}/lib/python

Run tests
^^^^^^^^^

**Fresnel** has extensive unit tests to verify correct execution.

.. code-block:: bash

   $ export PYTHONPATH=/path/to/build
   $ cd /path/to/fresnel
   $ pytest

Build user documentation
^^^^^^^^^^^^^^^^^^^^^^^^

Build the user documentation with **sphinx**:

.. code-block:: bash

   $ cd /path/to/fresnel
   $ cd doc
   $ make html
   $ open build/html/index.html

Specify search paths
^^^^^^^^^^^^^^^^^^^^

**OptiX**, **TBB**, **Embree**, and **Python** may be installed in a variety of locations. To specify locations
for libraries, use these methods the *first* time you invoke ``cmake`` in a clean build directory.

.. list-table::
   :header-rows: 1

   * - Library
     - Default search path
     - CMake Custom search path
   * - OptiX
     - ``/opt/optix``
     - ``-DOptiX_INSTALL_DIR=/path/to/optix``
   * - TBB
     - *system*
     - ``TBB_LINK=/path/to/tbb/lib`` (env var)
   * - Embree
     - *system*
     - ``-Dembree_DIR=/path/to/embree-3.x.y`` (the directory containing embree-config.cmake)
   * - Python
     - $PATH
     - ``-DPYTHON_EXECUTABLE=/path/to/bin/python``


Build C++ Documentation
^^^^^^^^^^^^^^^^^^^^^^^

To build the developer documentation, execute
``doxygen`` in the repository root. It will write HTML output in ``devdoc/html/index.html``.
