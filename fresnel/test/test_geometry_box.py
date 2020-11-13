"""Test the Box geometry."""

import os
import pathlib
from collections import namedtuple

import numpy
import PIL
import pytest

import conftest
import fresnel

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def scene_box(device):
    """Create a test scene."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    fresnel.geometry.Box(scene, [1, 2, 3, 0.4, 0.5, 0.6],
                         box_radius=0.2,
                         box_color=[1, 0, 1])

    scene.camera = fresnel.camera.Orthographic(position=(10, 10, 10),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=4)

    return scene


@pytest.fixture(scope="function")
def scene_box_(device_):
    """Pytest fixture to create a test scene."""
    return scene_box(device_)


def test_render(scene_box_, generate=False):
    """Test that Box renders properly."""
    buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
            open("output/test_geometry_box.test_render.png", "wb"), "png")
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / "reference" / "test_geometry_box.test_render.png")


def test_radius(scene_box_, generate=False):
    """Test the radius property."""
    geometry = scene_box_.geometry[0]

    r = numpy.array(0.1, dtype=numpy.float32)
    geometry.radius[:] = r
    numpy.testing.assert_array_equal(r, geometry.radius[:])

    buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
            open("output/test_geometry_box.test_radius.png", "wb"), "png")
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / "reference" / "test_geometry_box.test_radius.png")


def test_color(scene_box_, generate=False):
    """Test the color property."""
    geometry = scene_box_.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    c = numpy.zeros((12, 2, 3), dtype=numpy.float32) + [1, 0, 0]
    geometry.color[:] = c
    numpy.testing.assert_array_equal(c, geometry.color[:])

    buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
            open("output/test_geometry_box.test_color.png", "wb"), "png")
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / "reference" / "test_geometry_box.test_color.png")


def test_box_radius(scene_box_, generate=False):
    """Test the box_radius property."""
    geometry = scene_box_.geometry[0]

    r = 0.1
    geometry.box_radius = r
    assert r == pytest.approx(geometry.box_radius)

    buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
            open("output/test_geometry_box.test_box_radius.png", "wb"), "png")
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / "reference" / "test_geometry_box.test_box_radius.png")


def test_box_color(scene_box_, generate=False):
    """Test the box_color property."""
    geometry = scene_box_.geometry[0]

    c = numpy.array([1, 0, 1], dtype=numpy.float32)
    geometry.box_color = c
    numpy.testing.assert_array_equal(c, geometry.box_color)

    buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
            open("output/test_geometry_box.test_box_color.png", "wb"), "png")
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:],
            dir_path / "reference" / "test_geometry_box.test_box_color.png")


def test_box_update(scene_box_, generate=False):
    """Test the box property."""
    box_tuple = namedtuple("box_tuple", "Lx Ly Lz xy xz yz")
    box_list = [
        2,
        [1.5],
        [1, 2, 3],
        [1, 1.2, 2, 0.2, 0, 0.9],
        box_tuple(2, 2, 3, 0.5, 0.5, 0.7),
        {
            "Lx": 1,
            "Ly": 2,
            "Lz": 1,
            "xy": 0.4,
            "xz": 0.5,
            "yz": 0.3
        },
    ]

    for update_box in box_list:

        geometry = scene_box_.geometry[0]

        geometry.box = update_box

        assert isinstance(geometry.box, tuple)
        assert len(geometry.box) == 6

        buf_proxy = fresnel.preview(scene_box_, w=150, h=100, anti_alias=False)

        box_index = box_list.index(update_box)

        if generate:
            PIL.Image.fromarray(buf_proxy[:], mode="RGBA").save(
                open(f"output/test_geometry_box.test_render{box_index}.png",
                     "wb"), "png")
        else:
            conftest.assert_image_approx_equal(
                buf_proxy[:], dir_path / "reference"
                / f"test_geometry_box.test_render{box_index}.png")


if __name__ == "__main__":
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(("cpu", None)))

    scene = scene_box(device)
    test_render(scene, generate=True)

    scene = scene_box(device)
    test_radius(scene, generate=True)

    scene = scene_box(device)
    test_color(scene, generate=True)

    scene = scene_box(device)
    test_box_radius(scene, generate=True)

    scene = scene_box(device)
    test_box_color(scene, generate=True)

    scene = scene_box(device)
    test_box_update(scene, generate=True)
