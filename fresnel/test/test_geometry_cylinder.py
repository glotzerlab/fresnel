# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the Cylinder geometry."""

import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import pytest
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def scene_four_cylinders(device):
    """Create a test scene with four cylinders."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    position = [[[-5, -5, 0], [-5, 5, 0]], [[5, -5, -5], [5, 5, 5]],
                [[3, 3, -3], [-3, -3, -3]], [[-2, 2, 2], [2, -2, -2]]]

    fresnel.geometry.Cylinder(
        scene,
        points=position,
        radius=1.0,
        color=[0.9, 0.9, 0.9],
        material=fresnel.material.Material(
            color=fresnel.color.linear([0.42, 0.267, 1])),
    )

    scene.camera = fresnel.camera.Orthographic(position=(0, 2, 10),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=15)

    return scene


@pytest.fixture(scope='function')
def scene_four_cylinders_(device_):
    """Pytest fixture to create a test scene."""
    return scene_four_cylinders(device_)


def test_render(scene_four_cylinders_, generate=False):
    """Test that Cylinder renders properly."""
    buf_proxy = fresnel.preview(scene_four_cylinders_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_clyinder.test_render.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_clyinder.test_render.png')


def test_radius(scene_four_cylinders_, generate=False):
    """Test the radius property."""
    geometry = scene_four_cylinders_.geometry[0]

    r = numpy.array([0.5, 0.6, 0.8, 1.0], dtype=numpy.float32)
    geometry.radius[:] = r
    numpy.testing.assert_array_equal(r, geometry.radius[:])

    buf_proxy = fresnel.preview(scene_four_cylinders_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_clyinder.test_radius.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_clyinder.test_radius.png')


def test_points(scene_four_cylinders_, generate=False):
    """Test the points property."""
    geometry = scene_four_cylinders_.geometry[0]

    p = numpy.array([[[-5, -5, 0], [-5, 5, 0]], [[-3, 5, 0], [3, 5, 0]],
                     [[5, 5, 0], [5, -5, -0]], [[3, -5, 0], [-3, -5, 0]]],
                    dtype=numpy.float32)
    geometry.points[:] = p
    numpy.testing.assert_array_equal(p, geometry.points[:])

    buf_proxy = fresnel.preview(scene_four_cylinders_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_clyinder.test_position.png', 'wb'),
            'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_clyinder.test_position.png')


def test_color(scene_four_cylinders_, generate=False):
    """Test the color property."""
    geometry = scene_four_cylinders_.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    c = numpy.array(
        [[[0.9, 0, 0], [0, 0.9, 0]], [[0, 0, 0.9], [0, 0.9, 0.9]],
         [[0.9, 0.9, 0], [0.9, 0, 0.9]], [[0.1, 0.2, 0.9], [0.9, 0.2, 0.3]]],
        dtype=numpy.float32)
    geometry.color[:] = c
    numpy.testing.assert_array_equal(c, geometry.color[:])

    buf_proxy = fresnel.preview(scene_four_cylinders_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_clyinder.test_color.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_clyinder.test_color.png')


def test_outline(scene_four_cylinders_, generate=False):
    """Test that outlines render properly."""
    geometry = scene_four_cylinders_.geometry[0]
    geometry.outline_width = 0.3

    buf_proxy = fresnel.preview(scene_four_cylinders_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_clyinder.test_outline.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_clyinder.test_outline.png')


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = scene_four_cylinders(device)
    test_render(scene, generate=True)

    scene = scene_four_cylinders(device)
    test_radius(scene, generate=True)

    scene = scene_four_cylinders(device)
    test_points(scene, generate=True)

    scene = scene_four_cylinders(device)
    test_color(scene, generate=True)

    scene = scene_four_cylinders(device)
    test_outline(scene, generate=True)
