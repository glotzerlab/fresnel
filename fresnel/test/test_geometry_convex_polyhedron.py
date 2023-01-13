# Copyright (c) 2016-2023 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Test the ConvexPolyhedron geometry."""

import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import pytest
import math
import itertools
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent


def scene_eight_polyhedra(device):
    """Create a test scene with eight polyhedra."""
    scene = fresnel.Scene(device, lights=conftest.test_lights())

    # place eight polyhedra
    position = []
    for k in range(2):
        for i in range(2):
            for j in range(2):
                position.append([2.5 * i, 2.5 * j, 2.5 * k])

    # create the polyhedron faces
    origins = []
    normals = []
    colors = []

    for v in [-1, 1]:
        origins.append([v, 0, 0])
        normals.append([v, 0, 0])
        origins.append([0, v, 0])
        normals.append([0, v, 0])
        origins.append([0, 0, v])
        normals.append([0, 0, v])
        colors.append([178 / 255, 223 / 255, 138 / 255])
        colors.append([178 / 255, 223 / 255, 138 / 255])
        colors.append([178 / 255, 223 / 255, 138 / 255])

    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                normals.append([x, y, z])
                origins.append([x * 0.75, y * 0.75, z * 0.75])
                colors.append([166 / 255, 206 / 255, 227 / 255])

    poly_info = {
        'face_normal': normals,
        'face_origin': origins,
        'radius': math.sqrt(3),
        'face_color': fresnel.color.linear(colors)
    }
    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                 poly_info,
                                                 position=position)

    geometry.material = \
        fresnel.material.Material(color=fresnel.color.linear([1.0, 0, 0]),
                                  roughness=0.8,
                                  specular=0.5,
                                  primitive_color_mix=0.0)
    geometry.orientation[:] = [1, 0, 0, 0]

    scene.camera = fresnel.camera.Orthographic(position=(20, 20, 20),
                                               look_at=(0, 0, 0),
                                               up=(0, 1, 0),
                                               height=7)

    return scene


@pytest.fixture(scope='function')
def scene_eight_polyhedra_(device_):
    """Pytest fixture to create a test scene."""
    return scene_eight_polyhedra(device_)


def test_render(scene_eight_polyhedra_, generate=False):
    """Test that convex polyhedra render properly."""
    buf_proxy = fresnel.preview(scene_eight_polyhedra_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_convex_polyhedron.test_render.png',
                 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:], dir_path / 'reference'
            / 'test_geometry_convex_polyhedron.test_render.png')


def test_outline(scene_eight_polyhedra_, generate=False):
    """Test that face outlines render properly."""
    geometry = scene_eight_polyhedra_.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_eight_polyhedra_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_convex_polyhedron.test_outline.png',
                 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:], dir_path / 'reference'
            / 'test_geometry_convex_polyhedron.test_outline.png')


def test_face_color(scene_eight_polyhedra_, generate=False):
    """Test that faces can be colored individually."""
    buf_proxy = fresnel.preview(scene_eight_polyhedra_,
                                w=150,
                                h=100,
                                anti_alias=False)

    geometry = scene_eight_polyhedra_.geometry[0]
    geometry.color_by_face = 1.0
    geometry.material.primitive_color_mix = 1.0

    buf_proxy = fresnel.preview(scene_eight_polyhedra_,
                                w=150,
                                h=100,
                                anti_alias=False)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(
            open('output/test_geometry_convex_polyhedron.test_face_color.png',
                 'wb'), 'png')
    else:
        conftest.assert_image_approx_equal(
            buf_proxy[:], dir_path / 'reference'
            / 'test_geometry_convex_polyhedron.test_face_color.png')


def test_convert_cube():
    """Sanity checks on converting vertices to origins and normals."""
    pms = [+1, -1]
    cube_verts = numpy.array([x for x in itertools.product(pms, repeat=3)])

    poly_info = fresnel.util.convex_polyhedron_from_vertices(cube_verts)
    assert poly_info['face_origin'].shape[0] == 6
    assert poly_info['face_normal'].shape[0] == 6
    for f in poly_info['face_sides']:
        assert f == 4  # should all be squares
    assert poly_info['radius'] == numpy.sqrt(3)


def test_face_merge_cube():
    """Add a point into the middle and make sure no new faces are created."""
    pms = [+1, -1]
    cube_verts = numpy.array([x for x in itertools.product(pms, repeat=3)])

    cube_verts = numpy.concatenate((cube_verts, [[0.5, 0.5, 1.0]]))
    poly_info = fresnel.util.convex_polyhedron_from_vertices(cube_verts)
    assert poly_info['face_origin'].shape[0] == 6
    assert poly_info['face_normal'].shape[0] == 6
    for f in poly_info['face_sides']:
        assert f == 4  # should all be squares
    assert poly_info['radius'] == numpy.sqrt(3)


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = scene_eight_polyhedra(device)
    test_render(scene, generate=True)

    scene = scene_eight_polyhedra(device)
    test_outline(scene, generate=True)

    scene = scene_eight_polyhedra(device)
    test_face_color(scene, generate=True)
