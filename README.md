# fresnel

**fresnel** is a python library for path tracing publication quality images of soft matter simulations in real time.
The fastest render performance is possible on NVIDIA GPUs using their [OptiX](https://developer.nvidia.com/optix)
ray tracing engine. **fresnel** also supports multi-core CPUs using Intel's [Embree](https://embree.github.io/)
ray tracing kernels. Path tracing enables high quality global illumination and advanced rendering effects controlled by
intuitive parameters (like *roughness*, *specular*, and *metal*).

## fresnel Community

Use the [fresnel discussion board](https://github.com/glotzerlab/fresnel/discussions)
to post questions, ask for support, and discuss potential new features.
File bug reports on [fresnel's issue tracker](https://github.com/glotzerlab/fresnel/issues).
**fresnel** is an open source project. Please review
[the contributor's guide](CONTRIBUTING.md) for more information before contributing.

## Documentation

Read the [tutorial and reference documentation on readthedocs](https://fresnel.readthedocs.io/). The tutorial
is also available in Jupyter notebooks in the [fresnel-examples repository](https://github.com/glotzerlab/fresnel-examples).

## Gallery

Here are a few samples of what **fresnel** can do:

[<img alt="nature chemistry cover art" src="doc/gallery/protomer.png" width="290" />](https://www.nature.com/nchem/volumes/11/issues/3)
[<img alt="cuboid sample" src="doc/gallery/cuboid.png" width="290" />](doc/gallery/cuboid.py)
[<img alt="sphere sample" src="doc/gallery/sphere.png" width="290" />](doc/gallery/sphere.py)

## Installing fresnel

See the [installation guide](INSTALLING.rst) for details on installing fresnel with conda, docker, singularity,
and compiling from source.

## Example

This script generates the spheres gallery image:

```python
import fresnel, numpy, PIL

data = numpy.load('spheres.npz')

scene = fresnel.Scene()
scene.lights = fresnel.light.cloudy()

geometry = fresnel.geometry.Sphere(
    scene,
    position = data['position'],
    radius = 0.5,
    outline_width = 0.1)

geometry.material = fresnel.material.Material(
    color = fresnel.color.linear([0.1, 0.8, 0.1]),
    roughness = 0.8,
    specular = 0.2)

out = fresnel.pathtrace(scene, samples=64,
                        light_samples=32,
                        w=580, h=580)
PIL.Image.fromarray(out[:], mode='RGBA').save('sphere.png')
```
