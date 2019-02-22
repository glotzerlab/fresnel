Fresnel
++++++++++

**fresnel** is a python library for path tracing publication quality images of soft matter simulations in real time.
The fastest render performance is possible on NVIDIA GPUs using their `OptiX <https://developer.nvidia.com/optix>`_
ray tracing engine. **fresnel** also supports multi-core CPUs using Intel's `Embree <https://embree.github.io/>`_
ray tracing kernels. Path tracing enables high quality global illumination and advanced rendering effects.
**Fresnel** offers intuitive material parameters (like *roughness*, *specular*, and *metal*) and simple predefined
lighting setups (like *cloudy* and *lightbox*).

Here are a few samples of what **fresnel** can do. Click one of the gallery images below for a high resolution view and
a description of the **fresnel** features used to make it.

.. image:: gallery/protomer.png
    :width: 220px
    :alt: Protomer
    :target: gallery-cover-art.html#protomer

.. image:: gallery/cuboid.png
    :width: 220px
    :alt: Cuboids
    :target: gallery-features.html#cuboids

.. image:: gallery/sphere.png
    :width: 220px
    :alt: Spheres
    :target: gallery-features.html#spheres

.. toctree::
    :maxdepth: 1
    :caption: Gallery

    gallery-cover-art
    gallery-features

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

    examples/0*

.. toctree::
    :maxdepth: 1
    :caption: Primitives
    :glob:

    examples/1*

.. toctree::
    :maxdepth: 1
    :caption: Advanced topics
    :glob:

    examples/2*

.. toctree::
    :maxdepth: 1
    :caption: Interactive rendering
    :glob:

    examples/3*

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
