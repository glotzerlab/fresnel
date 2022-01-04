# Copyright (c) 2016-2022 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Cuboid example scene."""

import fresnel
import numpy
import PIL
import sys
import os

data = numpy.load('cuboids.npz')

scene = fresnel.Scene()
scene.lights = fresnel.light.lightbox()
W, H, D = data['width']
poly_info = fresnel.util.convex_polyhedron_from_vertices([
    [-W, -H, -D],
    [-W, -H, D],
    [-W, H, -D],
    [-W, H, D],
    [W, -H, -D],
    [W, -H, D],
    [W, H, -D],
    [W, H, D],
])

geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                             poly_info,
                                             position=data['position'],
                                             orientation=data['orientation'],
                                             outline_width=0.015)
geometry.material = fresnel.material.Material(color=fresnel.color.linear(
    [0.1, 0.1, 0.6]),
                                              roughness=0.1,
                                              specular=1)
geometry.outline_material = fresnel.material.Material(color=(0.95, 0.93, 0.88),
                                                      roughness=0.1,
                                                      metal=1.0)

scene.camera = fresnel.camera.Orthographic.fit(scene, view='front')

if 'CI' in os.environ:
    samples = 1
else:
    samples = 64

out = fresnel.pathtrace(scene, samples=samples, light_samples=32, w=580, h=580)
PIL.Image.fromarray(out[:], mode='RGBA').save('cuboid.png')

if len(sys.argv) > 1 and sys.argv[1] == 'hires':
    out = fresnel.pathtrace(scene,
                            samples=256,
                            light_samples=16,
                            w=1380,
                            h=1380)
    PIL.Image.fromarray(out[:], mode='RGBA').save('cuboid-hires.png')
