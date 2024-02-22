.. Copyright (c) 2016-2024 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Installing binaries
===================

**fresnel** binaries are available in the glotzerlab-software_ Docker_/Singularity_ images and in
packages on conda-forge_

.. _glotzerlab-software: https://glotzerlab-software.readthedocs.io
.. _Docker: https://hub.docker.com/
.. _Singularity: https://www.sylabs.io/
.. _conda-forge: https://conda-forge.org/docs/user/introduction.html

Singularity / Docker images
---------------------------

See the glotzerlab-software_ documentation for instructions to install and use the containers on
supported HPC clusters.

Conda package
-------------

**fresnel** is available on conda-forge_ on the *linux-64*, *osx-64*, and *osx-arm64* platforms.
Install the ``fresnel`` package from the conda-forge_ channel into a conda environment::

    $ conda install -c conda-forge fresnel

The fresnel builds on conda-forge_ support CPU rendering.

.. tip::

    Use mambaforge_, miniforge_ or miniconda_ instead of the full Anaconda distribution to avoid
    package conflicts with conda-forge_ packages.

.. _mambaforge: https://github.com/conda-forge/miniforge
.. _miniforge: https://github.com/conda-forge/miniforge
.. _miniconda: http://conda.pydata.org/miniconda.html
