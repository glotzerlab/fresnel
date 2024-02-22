# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Sphere example scene."""

import fresnel
import numpy
import PIL
import sys
import os

data = numpy.load('spheres.npz')

scene = fresnel.Scene()
scene.lights = fresnel.light.cloudy()

geometry = fresnel.geometry.Sphere(scene,
                                   position=data['position'],
                                   radius=0.5,
                                   outline_width=0.1)

geometry.material = fresnel.material.Material(color=fresnel.color.linear(
    [0.1, 0.8, 0.1]),
                                              roughness=0.8,
                                              specular=0.2)

scene.camera = fresnel.camera.Orthographic.fit(scene)

if 'CI' in os.environ:
    samples = 1
else:
    samples = 64

out = fresnel.pathtrace(scene, samples=samples, light_samples=32, w=580, h=580)
PIL.Image.fromarray(out[:], mode='RGBA').save('sphere.png')

if len(sys.argv) > 1 and sys.argv[1] == 'hires':
    out = fresnel.pathtrace(scene,
                            samples=256,
                            light_samples=16,
                            w=1380,
                            h=1380)
    PIL.Image.fromarray(out[:], mode='RGBA').save('sphere-hires.png')
