Change log
==========

`fresnel <https://github.com/glotzerlab/fresnel>`_ releases follow `semantic versioning <https://semver.org/>`_.

v0.7.0 (2019-02-08)
-------------------
* Add ``util.convex_polyhedron_from_vertices``: compute convex polyhedron plane origins and normals given a set of vertices
* Improve documentation
* Add ``interact.SceneView``: pyside2 widget for interactively rendering scenes with path tracing
* Add ``geometry.Mesh``: Arbitrary triangular mesh geometry, instanced with N positions and orientations
* **fresnel** development is now hosted on github: https://github.com/glotzerlab/fresnel/
* Improve ``light.lightbox`` lighting setup
* API breaking changes:
    * ``geometry.ConvexPolyhedron`` arguments changed. It now accepts polyhedron information as a dictionary.

v0.6.0 (2018-07-06)
-------------------

* Implement ``tracer.Path`` on the GPU.
* Implement ``ConvexPolyhedron`` geometry on the GPU.
* Improve path tracer performance with Russian roulette termination.
* Compile warning-free.
* Fix sphere intersection test bugs on the GPU.
* ``tracer.Path`` now correctly starts sampling over when resized.
* Wrap C++ code with pybind 2.2
* Make documentation available on readthedocs: http://fresnel.readthedocs.io
* Fresnel is now available on conda-forge: https://anaconda.org/conda-forge/fresnel
* embree >= 3.0 is now required for CPU support
* Improve documentation

v0.5.0 (2017-07-27)
-------------------

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
-------------------

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
-------------------

* Suppress "cannot import name" messages
* Support Nx3 and Nx4 inputs to ``color.linear``

v0.2.0 (2017-03-03)
-------------------

* Parallel rendering on the CPU
* Fix PTX file installation
* Fix python 2.7 support
* Unit tests
* Fix bug in sphere rendering on GPU

v0.1.0 (2017-02-02)
-------------------

* Prototype API
* Sphere geometry
* Prism geometry
* outline materials
* diffuse materials
* Direct tracer
