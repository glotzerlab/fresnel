# Copyright (c) 2016-2021 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the Sphere geometry."""

import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import pytest
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def scene_four_spheres(device):
    """Create a test scene with four spheres."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    mat = fresnel.material.Material(
        color=fresnel.color.linear([0.42, 0.267, 1]))
    fresnel.geometry.Sphere(scene,
                            position=[[1, 0, 1], [1, 0, -1], [-1, 0, 1],
                                      [-1, 0, -1]],
                            radius=1.0,
                            material=mat,
                            color=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])

    scene.camera = fresnel.camera.Orthographic(position=(10, 10, 10),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=4)

    return scene


@pytest.fixture(scope='function')
def scene_four_spheres_(device_):
    """Pytest fixture to create a test scene."""
    return scene_four_spheres(device_)


def test_render(scene_four_spheres_, generate=False):
    """Test that spheres render properly."""
    buf_proxy = fresnel.preview(scene_four_spheres_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_sphere.test_render.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_sphere.test_render.png')


def test_radius(scene_four_spheres_, generate=False):
    """Test the radius property."""
    geometry = scene_four_spheres_.geometry[0]

    r = numpy.array([0.5, 0.6, 0.8, 1.0], dtype=numpy.float32)
    geometry.radius[:] = r
    numpy.testing.assert_array_equal(r, geometry.radius[:])

    buf_proxy = fresnel.preview(scene_four_spheres_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_sphere.test_radius.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_sphere.test_radius.png')


def test_position(scene_four_spheres_, generate=False):
    """Test the position property."""
    geometry = scene_four_spheres_.geometry[0]

    p = numpy.array([[1.5, 0, 1], [1.5, 0, -1], [-1.5, 0, 1], [-1.5, 0, -1]],
                    dtype=numpy.float32)
    geometry.position[:] = p
    numpy.testing.assert_array_equal(p, geometry.position[:])

    buf_proxy = fresnel.preview(scene_four_spheres_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_sphere.test_position.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_sphere.test_position.png')


def test_color(scene_four_spheres_, generate=False):
    """Test the color property."""
    geometry = scene_four_spheres_.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    c = fresnel.color.linear(
        numpy.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                    dtype=numpy.float32))
    geometry.color[:] = c
    numpy.testing.assert_array_equal(c, geometry.color[:])

    buf_proxy = fresnel.preview(scene_four_spheres_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_sphere.test_color.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_sphere.test_color.png')


def test_outline(scene_four_spheres_, generate=False):
    """Test that outlines render properly."""
    geometry = scene_four_spheres_.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_four_spheres_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_sphere.test_outline.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_sphere.test_outline.png')


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = scene_four_spheres(device)
    test_render(scene, generate=True)

    scene = scene_four_spheres(device)
    test_radius(scene, generate=True)

    scene = scene_four_spheres(device)
    test_position(scene, generate=True)

    scene = scene_four_spheres(device)
    test_color(scene, generate=True)

    scene = scene_four_spheres(device)
    test_outline(scene, generate=True)
