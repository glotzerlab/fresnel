# Fresnel Change Log

[TOC]

## v0.4.0

Not yet released

* Enforce requirement: Embree >= 2.10.0
* Enforce requirement Pybind =1.8.1
* Enforce requirement TBB >= 4.3
* Rewrite camera API, add camera.fit to fit the scene
* scenes default to an automatic fit camera
* API breaking changes:
    * ``camera.Orthographic`` is now ``camera.orthographic``

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
