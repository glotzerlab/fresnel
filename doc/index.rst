.. Copyright (c) 2016-2023 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Fresnel
++++++++++

**fresnel** is a python library for path tracing publication quality images of soft matter simulations in real time.
The fastest render performance is possible on NVIDIA GPUs using their `OptiX <https://developer.nvidia.com/optix>`_
ray tracing engine. **fresnel** also supports multi-core CPUs using Intel's `Embree <https://embree.github.io/>`_
ray tracing kernels. Path tracing enables high quality global illumination and advanced rendering effects.
**Fresnel** offers intuitive material parameters (like *roughness*, *specular*, and *metal*) and simple predefined
lighting setups (like *cloudy* and *lightbox*).

Here are a few samples of what **fresnel** can do:

.. image:: gallery/protomer.png
    :width: 220px
    :alt: Protomer
    :target: gallery/protomer.html

.. image:: gallery/cuboid.png
    :width: 220px
    :alt: Cuboids
    :target: gallery/cuboid.html

.. image:: gallery/sphere.png
    :width: 220px
    :alt: Spheres
    :target: gallery/sphere.html

.. toctree::
    :maxdepth: 1
    :caption: Examples

    gallery

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    building
    changes
    community

.. toctree::
    :maxdepth: 1
    :caption: Basic tutorials
    :glob:

    examples/00-Basic-tutorials/*

.. toctree::
    :maxdepth: 1
    :caption: Primitives
    :glob:

    examples/01-Primitives/*

.. toctree::
    :maxdepth: 1
    :caption: Advanced topics
    :glob:

    examples/02-Advanced-topics/*

.. toctree::
    :maxdepth: 2
    :caption: Python API Reference

    package-fresnel

.. toctree::
    :maxdepth: 1
    :caption: Developer guide

    contributing
    style
    testing

.. toctree::
    :maxdepth: 1
    :caption: Additional information

    license
    credits
    indices
