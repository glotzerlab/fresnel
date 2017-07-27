# Fresnel Change Log

[TOC]

## v0.5.0

Released 2017/07/27

* Add new lighting setups
    * `lightbox`
    * `cloudy`
    * `ring`
* Adjust brightness of lights in existing setups
* Remove `clearcoat` material parameter
* Add `spec_trans` material parameter
* Add `Path` tracer to render scenes with indirect lighting, reflections, and transparency (*CPU-only*)
* Add `ConvexPolyhedron` geometry (*CPU-only*, *beta API, subject to change*)
* Add `fresnel.preview` function to easily generate `Preview` traced renders with one line
* Add `fresnel.pathtrace` function to easily generate `Path` traced renders with one line
* Add anti-aliasing (always on for the `Path` tracer, `set aa_level` > 0 to enable for `Preview`)
* API breaking changes:
    * `render` no longer exists. Use `preview` or `pathtrace`.
    * `tracer.Direct` is now `tracer.Preview`.

CPU-only features will be implemented on the GPU in a future release.

## v0.4.0

Released 2017/04/03

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

## v0.3.0

Released 2017/03/09

* Suppress "cannot import name" messages
* Support Nx3 and Nx4 inputs to ``color.linear``

## v0.2.0

* Parallel rendering on the CPU
* Fix PTX file installation
* Fix python 2.7 support
* Unit tests
* Fix bug in sphere rendering on GPU

## v0.1.0

* Prototype API
* Sphere geometry
* Prism geometry
* outline materials
* diffuse materials
* Direct tracer
