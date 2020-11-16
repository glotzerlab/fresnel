"""Test the Polygon geometry."""

import fresnel
import numpy
from collections import namedtuple
import PIL
import pytest
import conftest
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def scene_rounded_polygons(device):
    """Create a test scene with polygons."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    mat = fresnel.material.Material(color=fresnel.color.linear([0.42, 0.267,
                                                                1]),
                                    solid=1)

    fresnel.geometry.Polygon(scene,
                             N=2,
                             rounding_radius=0.3,
                             vertices=[[-1, -1], [1, -1], [1, 1], [0, 0],
                                       [-1, 1]],
                             position=[[-1.5, 0], [1.5, 0]],
                             angle=[0.1, -0.2],
                             color=[[0, 0, 1], [0, 1, 0]],
                             material=mat)

    scene.camera = fresnel.camera.Orthographic(position=(0, 0, -2),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=5)

    return scene


@pytest.fixture(scope='function')
def scene_rounded_polygons_(device_):
    """Pytest fixture to create test scene."""
    return scene_rounded_polygons(device_)


def scene_polygons(device):
    """Create a test scene with polygons."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    mat = fresnel.material.Material(color=fresnel.color.linear([0.42, 0.267,
                                                                1]),
                                    solid=1)

    fresnel.geometry.Polygon(scene,
                             N=2,
                             rounding_radius=0,
                             vertices=[[-1, -1], [1, -1], [1, 1], [0, 0],
                                       [-1, 1]],
                             position=[[-1.5, 0], [1.5, 0]],
                             angle=[0.1, -0.2],
                             color=[[0, 0, 1], [0, 1, 0]],
                             material=mat)

    scene.camera = fresnel.camera.Orthographic(position=(0, 0, -2),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=5)

    return scene


@pytest.fixture(scope='function')
def scene_polygons_(device_):
    """Pytest fixture to create test scene."""
    return scene_polygons(device_)


def test_render(scene_polygons_, generate=False):
    """Test that Polygons render properly."""
    buf_proxy = fresnel.preview(scene_polygons_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_render.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_render.png')


def test_rounded(scene_rounded_polygons_, generate=False):
    """Test that rounded polygons render properly."""
    geometry = scene_rounded_polygons_.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_rounded_polygons_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_rounded.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_rounded.png')


def test_angle(scene_polygons_, generate=False):
    """Test that polygons can be rotated."""
    geometry = scene_polygons_.geometry[0]

    a = numpy.array([-0.8, 0.5], dtype=numpy.float32)
    geometry.angle[:] = a
    numpy.testing.assert_array_equal(a, geometry.angle[:])

    buf_proxy = fresnel.preview(scene_polygons_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_angle.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_angle.png')


def test_position(scene_polygons_, generate=False):
    """Test the position property."""
    geometry = scene_polygons_.geometry[0]

    p = numpy.array([[-2, 0], [3, 0]], dtype=numpy.float32)
    geometry.position[:] = p
    numpy.testing.assert_array_equal(p, geometry.position[:])

    buf_proxy = fresnel.preview(scene_polygons_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_position.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_position.png')


def test_color(scene_polygons_, generate=False):
    """Test the color property."""
    geometry = scene_polygons_.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    c = numpy.array(
        [fresnel.color.linear([0, 1, 0]),
         fresnel.color.linear([1, 0, 0])],
        dtype=numpy.float32)
    geometry.color[:] = c
    numpy.testing.assert_array_equal(c, geometry.color[:])

    buf_proxy = fresnel.preview(scene_polygons_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_color.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_color.png')


def test_outline(scene_polygons_, generate=False):
    """Test that outlines render properly."""
    geometry = scene_polygons_.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_polygons_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_polygon.test_outline.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_geometry_polygon.test_outline.png')


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = scene_polygons(device)
    test_render(scene, generate=True)

    scene = scene_rounded_polygons(device)
    test_rounded(scene, generate=True)

    scene = scene_polygons(device)
    test_angle(scene, generate=True)

    scene = scene_polygons(device)
    test_position(scene, generate=True)

    scene = scene_polygons(device)
    test_color(scene, generate=True)

    scene = scene_polygons(device)
    test_outline(scene, generate=True)
