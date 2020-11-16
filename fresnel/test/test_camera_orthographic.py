"""Test the Orthographic camera."""

import fresnel
import pytest
import numpy


def test_camera_fit_no_geometry(device_):
    """Test that fit() errors when there is no geometry."""
    scene = fresnel.Scene()

    # fit cannot be called on a scene with no geometries
    with pytest.raises(ValueError):
        fresnel.camera.Orthographic.fit(scene, view='front', margin=0)


def test_camera_fit_front(device_):
    """Test that fit works with the front option."""
    scene = fresnel.Scene()

    fresnel.geometry.Sphere(scene,
                            position=[[-9, -2, 0], [-5, -1, 0], [4, 0, 0],
                                      [2, 1, 0]],
                            radius=1.0)

    cam = fresnel.camera.Orthographic.fit(scene, view='front', margin=0)
    assert cam.position[0] == -2.5
    assert cam.position[1] == -0.5
    numpy.testing.assert_array_equal(cam.look_at, (-2.5, -0.5, 0))
    assert cam.height == 5


def test_camera_fit_isometric(device_):
    """Test that fit works with the isometric option."""
    scene = fresnel.Scene()

    fresnel.geometry.Sphere(scene,
                            position=[[-9, -2, 0], [-5, -1, 0], [4, 0, 0],
                                      [2, 1, 0]],
                            radius=1.0)

    fresnel.camera.Orthographic.fit(scene, view='isometric', margin=0)
    # isometric cameras do not have a simple testable format, just test that the
    # API works


def test_scene_default_camera(device_):
    """Test that there is a default camera."""
    scene = fresnel.Scene()

    fresnel.geometry.Sphere(scene,
                            position=[[-9, -2, 0], [-5, -1, 0], [4, 0, 0],
                                      [2, 1, 0]],
                            radius=1.0)

    fresnel.preview(scene, anti_alias=False)


def test_orthographic_attributes():
    """Test all properties."""
    cam = fresnel.camera.Orthographic(position=(1, 0, 10),
                                      look_at=(1, 0, 0),
                                      up=(0, 1, 0),
                                      height=6)
    numpy.testing.assert_array_equal(cam.position, (1, 0, 10))
    numpy.testing.assert_array_equal(cam.look_at, (1, 0, 0))
    numpy.testing.assert_array_equal(cam.up, (0, 1, 0))
    assert cam.height == 6

    cam2 = eval(repr(cam))
    numpy.testing.assert_array_equal(cam2.position, (1, 0, 10))
    numpy.testing.assert_array_equal(cam2.look_at, (1, 0, 0))
    numpy.testing.assert_array_equal(cam2.up, (0, 1, 0))
    assert cam2.height == 6

    cam.position = (1, 3, 8)
    cam.look_at = (20, 5, 18)
    cam.up = (1, 2, 3)
    cam.height = 112

    numpy.testing.assert_array_equal(cam.position, (1, 3, 8))
    numpy.testing.assert_array_equal(cam.look_at, (20, 5, 18))
    numpy.testing.assert_array_equal(cam.up, (1, 2, 3))
    assert cam.height == 112
