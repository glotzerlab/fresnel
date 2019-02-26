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
    :target: gallery-research.html#protomer

.. image:: gallery/cuboid.png
    :width: 220px
    :alt: Cuboids
    :target: gallery-features.html#cuboids

.. image:: gallery/sphere.png
    :width: 220px
    :alt: Spheres
    :target: gallery-features.html#spheres

Gallery
++++++++++

.. toctree::
    :maxdepth: 1
    :caption: Gallery

    gallery-research
    gallery-features
    gallery-alternate

Getting started
+++++++++++++++

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
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
    :maxdepth: 3
    :caption: Python API Reference

    package-fresnel

.. toctree::
    :maxdepth: 1
    :caption: Additional information

    license
    credits
    indices
