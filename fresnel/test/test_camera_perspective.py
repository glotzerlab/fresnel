# Copyright (c) 2016-2022 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the Perspective camera."""

import fresnel
import numpy
from collections import namedtuple
import conftest
import pytest


def scene(device):
    """Create a test scene."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    position = []
    for i in range(-8, 8):
        for j in range(-8, 8):
            position.append([i, 0, j])

    mat = fresnel.material.Material(
        color=fresnel.color.linear([0.42, 0.267, 0.9]))
    fresnel.geometry.Sphere(
        scene,
        position=position,
        radius=0.7,
        material=mat,
    )

    scene.camera = fresnel.camera.Perspective(position=(0, 4, -10),
                                              look_at=(0, 0, 0),
                                              up=(0, 1, 0),
                                              focus_distance=10)

    return scene


@pytest.fixture(scope='function')
def scene_(device_):
    """Pytest fixture to create a test scene."""
    return scene(device_)


def test_perspective_attributes():
    """Test all properties."""
    cam = fresnel.camera.Perspective(position=(0, 0, -10),
                                     look_at=(0, 0, 0),
                                     up=(0, 1, 0),
                                     height=0.25,
                                     focal_length=1.25,
                                     focus_distance=20,
                                     f_stop=1.5)
    numpy.testing.assert_array_equal(cam.position, (0, 0, -10))
    numpy.testing.assert_array_equal(cam.look_at, (0, 0, 0))
    numpy.testing.assert_array_equal(cam.up, (0, 1, 0))
    assert cam.height == 0.25
    assert cam.focal_length == 1.25
    assert cam.focus_distance == 20
    assert cam.f_stop == 1.5

    cam2 = eval(repr(cam))
    numpy.testing.assert_array_equal(cam2.position, (0, 0, -10))
    numpy.testing.assert_array_equal(cam2.look_at, (0, 0, 0))
    numpy.testing.assert_array_equal(cam2.up, (0, 1, 0))
    assert cam2.height == 0.25
    assert cam2.focal_length == 1.25
    assert cam2.focus_distance == 20
    assert cam2.f_stop == 1.5

    numpy.testing.assert_allclose(cam.depth_of_field, 0.225007119365887)
    cam.depth_of_field = 1
    numpy.testing.assert_allclose(cam.depth_of_field, 1)
    numpy.testing.assert_allclose(cam.f_stop, 6.662505149841309)

    numpy.testing.assert_allclose(cam.focus_on, [0, 0, 10])
    cam.focus_on = [20, -50, 5]
    numpy.testing.assert_allclose(cam.focus_on, [0, 0, 5])
    numpy.testing.assert_allclose(cam.focus_distance, 15)

    numpy.testing.assert_allclose(cam.vertical_field_of_view,
                                  0.19933730498232408)
    cam.vertical_field_of_view = 0.4
    numpy.testing.assert_allclose(cam.focal_length, 0.6166443824768066)


def test_render(scene_, generate=False):
    """Test that preview can render a scene."""
    conftest.check_preview_render(scene_,
                                  "test_camera_perspective.test_render",
                                  generate=generate)


def test_focal_length(scene_, generate=False):
    """Test the focal_length property."""
    scene_.camera.focal_length = 0.25

    conftest.check_preview_render(scene_,
                                  "test_camera_perspective.test_focal_length",
                                  generate=generate)


def test_height(scene_, generate=False):
    """Test the height property."""
    scene_.camera.height = .125

    conftest.check_preview_render(scene_,
                                  "test_camera_perspective.test_height",
                                  generate=generate)


def test_pathtrace(scene_, generate=False):
    """Test that Preview works with pathtrace."""
    conftest.check_pathtrace_render(scene_,
                                    "test_camera_perspective.test_pathtrace",
                                    generate=generate)


def test_f_stop(scene_, generate=False):
    """Test the f_stop property."""
    scene_.camera.f_stop = 0.2

    conftest.check_pathtrace_render(scene_,
                                    "test_camera_perspective.test_f_stop",
                                    generate=generate)


def test_focus_distance(scene_, generate=False):
    """Test the focus_distance property."""
    scene_.camera.f_stop = 0.2
    scene_.camera.focus_distance = 6

    conftest.check_pathtrace_render(
        scene_,
        "test_camera_perspective.test_focal_distance",
        generate=generate)


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    for method in [
            test_render,
            test_focal_length,
            test_height,
            test_pathtrace,
            test_f_stop,
            test_focus_distance,
    ]:
        s = scene(device)
        method(s, generate=True)
