.. Copyright (c) 2016-2021 The Regents of the University of Michigan
.. Part of fresnel, released under the BSD 3-Clause License.

Change log
==========

fresnel_ releases follow `semantic versioning`_.

.. _fresnel:  https://github.com/glotzerlab/fresnel
.. _semantic versioning: https://semver.org/

v0.x
----

v0.13.4 (2021-11-17)
^^^^^^^^^^^^^^^^^^^^

*Added*

* Support Python 3.10.
* Support clang 13.

v0.13.3 (2021-06-07)
^^^^^^^^^^^^^^^^^^^^

*Added*

* Support Windows.

v0.13.2 (2021-05-11)
^^^^^^^^^^^^^^^^^^^^

*Added*

* Support macos-arm64.

v0.13.1 (2021-03-11)
^^^^^^^^^^^^^^^^^^^^

*Fixed*

* Add missing ``version`` module

v0.13.0 (2021-03-11)
^^^^^^^^^^^^^^^^^^^^

*Added*

* Perspective camera.
* Depth of field effect.

*Changed*

* Reduce latency in ``interact.SceneView`` while rotating the view.
* Improve user experience with mouse rotations in ``interact.SceneView``.
* [breaking] - Moved ``camera.orthographic`` to ``camera.Orthographic``.
* [breaking] - Moved ``camera.fit`` to ``camera.Orthographic.fit``.

*Removed*

* [breaking] - Removed "auto" camera in ``Scene``. Use
  ``camera.Orthographic.fit``

v0.12.0 (2020-02-27)
^^^^^^^^^^^^^^^^^^^^

*Added*

* ``preview`` and ``tracer.Preview`` accept a boolean flag ``anti_alias`` to
  enable or disable anti-aliasing.

*Changed*

* ``preview`` and ``tracer.Preview`` enable anti-alisasing by default.
* Python, Cython, and C code must follow strict style guidelines.
* Renamed ``util.array`` to ``util.Array``
* Renamed ``util.image_array`` to ``util.ImageArray``
* Converted ``interact.SceneView.setScene`` to a property: ``scene``

*Removed*

* ``preview`` and ``tracer.Preview`` no longer accept the ``aa_level`` argument
  - use ``anti_alias``.

v0.11.0 (2019-10-30)
^^^^^^^^^^^^^^^^^^^^

*Added*

*  Added box geometry convenience class ``Box``.

*Removed*

* Support for **Python** 3.5.

*Fixed*

* Compile on systems where ``libqhullcpp.a`` is missing or broken.
* Find **Embree** headers when they are not in the same path as **TBB**.

v0.10.1 (2019-09-05)
^^^^^^^^^^^^^^^^^^^^

*Fixed*

* Restore missing examples on readthedocs.

v0.10.0 (2019-08-19)
^^^^^^^^^^^^^^^^^^^^

*Changed*

* **CMake** >= 3.8 is required at build time.
* **pybind11** >= 2.2 is required at build time.
* **qhull** >= 2015 is required.
* install to the **Python** ``site-packages`` directory by default.
* **CI** tests execute on Microsoft Azure Pipelines.

*Fixed*

* Improved installation documentation.


v0.9.0 (2019-04-30)
^^^^^^^^^^^^^^^^^^^

* Added support for linearizing colors of shape (4,).
* Improve examples.

v0.8.0 (2019-03-05)
^^^^^^^^^^^^^^^^^^^

* Documentation improvements.
* Add ``geometry.Polygon``: Simple and/or rounded polygons in the *z=0* plane.
* API breaking changes:

  * Remove: ``geometry.Prism``

v0.7.1 (2019-02-05)
^^^^^^^^^^^^^^^^^^^

* Fix **conda-forge** build on mac

v0.7.0 (2019-02-05)
^^^^^^^^^^^^^^^^^^^
* Add ``util.convex_polyhedron_from_vertices``: compute convex polyhedron plane origins and normals given a set of vertices
* Improve documentation
* Add ``interact.SceneView``: **pyside2** widget for interactively rendering scenes with path tracing
* Add ``geometry.Mesh``: Arbitrary triangular mesh geometry, instanced with N positions and orientations
* **fresnel** development is now hosted on github: https://github.com/glotzerlab/fresnel/
* Improve ``light.lightbox`` lighting setup
* API breaking changes:

  * ``geometry.ConvexPolyhedron`` arguments changed. It now accepts polyhedron information as a dictionary.

v0.6.0 (2018-07-06)
^^^^^^^^^^^^^^^^^^^

* Implement ``tracer.Path`` on the GPU.
* Implement ``ConvexPolyhedron`` geometry on the GPU.
* Improve path tracer performance with Russian roulette termination.
* Compile warning-free.
* Fix sphere intersection test bugs on the GPU.
* ``tracer.Path`` now correctly starts sampling over when resized.
* Wrap C++ code with **pybind** 2.2
* Make documentation available on readthedocs: http://fresnel.readthedocs.io
* Fresnel is now available on **conda-forge**: https://anaconda.org/conda-forge/fresnel
* embree >= 3.0 is now required for CPU support
* Improve documentation

v0.5.0 (2017-07-27)
^^^^^^^^^^^^^^^^^^^

* Add new lighting setups

  * ``lightbox``
  * ``cloudy``
  * ``ring``

* Adjust brightness of lights in existing setups
* Remove ``clearcoat`` material parameter
* Add ``spec_trans`` material parameter
* Add ``Path`` tracer to render scenes with indirect lighting, reflections, and transparency (\ *CPU-only*\ )
* Add ``ConvexPolyhedron`` geometry (\ *CPU-only*\ , *beta API, subject to change*\ )
* Add ``fresnel.preview`` function to easily generate ``Preview`` traced renders with one line
* Add ``fresnel.pathtrace`` function to easily generate ``Path`` traced renders with one line
* Add anti-aliasing (always on for the ``Path`` tracer, ``set aa_level`` > 0 to enable for ``Preview``\ )
* API breaking changes:

  * ``render`` no longer exists. Use ``preview`` or ``pathtrace``.
  * ``tracer.Direct`` is now ``tracer.Preview``.

CPU-only features will be implemented on the GPU in a future release.

v0.4.0 (2017-04-03)
^^^^^^^^^^^^^^^^^^^

* Enforce requirement: Embree >= 2.10.0
* Enforce requirement Pybind =1.8.1
* Enforce requirement TBB >= 4.3
* Rewrite camera API, add camera.fit to fit the scene
* scenes default to an automatic fit camera
* Implement area lights, add default lighting setups
* ``Scene`` now supports up to 4 lights, specified in camera space
* Implement Disney's principled BRDF
* ``Tracer.histogram`` computes a histogram of the rendered image
* ``Tracer.enable_highlight_warning`` highlights overexposed pixels with a given warning color
* ``Device.available_modes`` lists the available execution modes
* ``Device.available_gpus`` lists the available GPUs
* ``Device`` can now be limited to *n* GPUs
* API breaking changes:

  * ``camera.Orthographic`` is now ``camera.orthographic``
  * ``Device`` now takes the argument *n* instead of *limit*
  * ``Scene`` no longer has a ``light_direction`` member

v0.3.0 (2017-03-09)
^^^^^^^^^^^^^^^^^^^

* Suppress "cannot import name" messages
* Support Nx3 and Nx4 inputs to ``color.linear``

v0.2.0 (2017-03-03)
^^^^^^^^^^^^^^^^^^^

* Parallel rendering on the CPU
* Fix PTX file installation
* Fix python 2.7 support
* Unit tests
* Fix bug in sphere rendering on GPU

v0.1.0 (2017-02-02)
^^^^^^^^^^^^^^^^^^^

* Prototype API
* Sphere geometry
* Prism geometry
* outline materials
* diffuse materials
* Direct tracer
