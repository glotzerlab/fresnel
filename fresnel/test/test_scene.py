"""Test the Scene class."""

import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def test_background_color(device_):
    """Test the background_color property."""
    scene = fresnel.Scene(device=device_)

    scene.background_color = fresnel.color.linear((0.125, 0.75, 0.375))

    numpy.testing.assert_array_equal(scene.background_color,
                                     fresnel.color.linear((0.125, 0.75, 0.375)))

    scene.background_alpha = 0.5

    assert scene.background_alpha == 0.5

    scene.camera = fresnel.camera.Orthographic(position=(0, 0, 10),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=7)
    buf_proxy = fresnel.preview(scene, w=100, h=100, anti_alias=False)
    buf = buf_proxy[:]

    numpy.testing.assert_array_equal(
        buf[:, :, 3],
        numpy.ones(shape=(100, 100), dtype=buf.dtype) * 128)
    numpy.testing.assert_array_equal(
        buf[:, :, 0:3],
        numpy.ones(shape=(100, 100, 3), dtype=buf.dtype) * (32, 191, 96))


def test_camera(scene_hex_sphere_, generate=False):
    """Test the camera property."""
    scene_hex_sphere_.camera = fresnel.camera.Orthographic(position=(1, 0, 10),
                                                           look_at=(1, 0, 0),
                                                           up=(0, 1, 0),
                                                           height=6)

    numpy.testing.assert_array_equal(scene_hex_sphere_.camera.position,
                                     (1, 0, 10))
    numpy.testing.assert_array_equal(scene_hex_sphere_.camera.look_at,
                                     (1, 0, 0))
    numpy.testing.assert_array_equal(scene_hex_sphere_.camera.up, (0, 1, 0))
    assert scene_hex_sphere_.camera.height == 6

    buf_proxy = fresnel.preview(scene_hex_sphere_,
                                w=100,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_camera.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:], dir_path / 'reference' / 'test_scene.test_camera.png')


def test_light_dir(scene_hex_sphere_, generate=False):
    """Test the lights property."""
    scene_hex_sphere_.lights[0].direction = (1, 0, 0)
    scene_hex_sphere_.lights[0].color = (0.5, 0.5, 0.5)
    assert scene_hex_sphere_.lights[0].direction == (1, 0, 0)
    assert scene_hex_sphere_.lights[0].color == (0.5, 0.5, 0.5)

    buf_proxy = fresnel.preview(scene_hex_sphere_,
                                w=100,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_light_dir.png', 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_scene.test_light_dir.png')


def test_multiple_geometries(device_, generate=False):
    """Test multiple geometries."""
    scene = fresnel.Scene(lights=conftest.test_lights())
    scene.camera = fresnel.camera.Orthographic(position=(0, 0, 10),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=7)

    geom1 = fresnel.geometry.Sphere(scene,
                                    position=[[-4, 1, 0], [-4, -1, 0],
                                              [-2, 1, 0], [-2, -1, 0]],
                                    radius=1.0)
    geom1.material = fresnel.material.Material(solid=1.0,
                                               color=fresnel.color.linear(
                                                   [0.42, 0.267, 1]))
    geom1.outline_width = 0.12

    geom2 = fresnel.geometry.Sphere(scene,
                                    position=[[4, 1, 0], [4, -1, 0], [2, 1, 0],
                                              [2, -1, 0]],
                                    radius=1.0)
    geom2.material = fresnel.material.Material(solid=0.0,
                                               color=fresnel.color.linear(
                                                   [1, 0.874, 0.169]))

    buf_proxy = fresnel.preview(scene, w=200, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_multiple_geometries1.png', 'wb'),
            'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_scene.test_multiple_geometries1.png')

    geom1.disable()

    buf_proxy = fresnel.preview(scene, w=200, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_multiple_geometries2.png', 'wb'),
            'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_scene.test_multiple_geometries2.png')

    geom1.enable()

    buf_proxy = fresnel.preview(scene, w=200, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_multiple_geometries3.png', 'wb'),
            'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_scene.test_multiple_geometries3.png')

    geom2.remove()

    buf_proxy = fresnel.preview(scene, w=200, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_scene.test_multiple_geometries4.png', 'wb'),
            'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / 'reference' / 'test_scene.test_multiple_geometries4.png')


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = conftest.scene_hex_sphere(device)
    test_camera(scene, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_light_dir(scene, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_multiple_geometries(scene, generate=True)
