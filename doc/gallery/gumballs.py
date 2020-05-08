"""Gumballs example scene."""

import fresnel
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import PIL

# First, we create a color map for gumballs.
colors = [
    '#e56d60',
    '#ee9944',
    '#716e80',
    '#eadecd',
    '#cec746',
    '#c0443f',
    '#734d56',
    '#5d5f7b',
    '#ecb642',
    '#8a9441']
cmap = LinearSegmentedColormap.from_list(name='gumball',
                                         colors=colors,
                                         N=len(colors))

# Next, we gather information needed for the geometry.
position = np.load('gumballs.npz')['position']
np.random.seed(123)
color = fresnel.color.linear(cmap(np.random.rand(len(position))))
material = fresnel.material.Material(primitive_color_mix=1.0,
                                     roughness=0.2,
                                     specular=0.8,
                                     )

# We create a fresnel scene and its geometry.
scene = fresnel.Scene()

geometry = fresnel.geometry.Sphere(scene,
                                   position=position,
                                   radius=0.5,
                                   color=color,
                                   material=material,
                                   )

# Configure camera and lighting.
scene.camera = fresnel.camera.fit(scene, view='front')
scene.camera.height = 10
scene.lights = fresnel.light.lightbox()
scene.lights.append(
    fresnel.light.Light(direction=(0.3, -0.3, 1),
                        color=(0.5, 0.5, 0.5),
                        theta=np.pi))

# Execute rendering.
out = fresnel.pathtrace(scene, w=600, h=600, samples=128, light_samples=64)
PIL.Image.fromarray(out[:], mode='RGBA').save('gumballs.png')

out = fresnel.pathtrace(scene, w=1500, h=1500, samples=256, light_samples=64)
PIL.Image.fromarray(out[:], mode='RGBA').save('gumballs-hires.png')
