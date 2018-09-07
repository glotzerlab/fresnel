import fresnel
import itertools
import numpy
from collections import namedtuple
import PIL
import conftest


def test_render(scene_eight_polyhedra, generate=False):
    buf_proxy = fresnel.preview(scene_eight_polyhedra, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_convex_polyhedron.test_render.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_convex_polyhedron.test_render.png')


def test_outline(scene_eight_polyhedra, generate=False):
    geometry = scene_eight_polyhedra.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_eight_polyhedra, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_convex_polyhedron.test_outline.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_convex_polyhedron.test_outline.png')


def test_face_color(scene_eight_polyhedra, generate=False):
    buf_proxy = fresnel.preview(scene_eight_polyhedra, w=150, h=100)

    geometry = scene_eight_polyhedra.geometry[0]
    geometry.color_by_face = 1.0
    geometry.material.primitive_color_mix = 1.0

    buf_proxy = fresnel.preview(scene_eight_polyhedra, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_convex_polyhedron.test_face_color.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_convex_polyhedron.test_face_color.png')


def test_face_merge_cube(cube_verts):
    origins, normals, r = fresnel.util.convex_polyhedron_from_vertices(cube_verts)
    assert origins.shape[0] == 6
    assert normals.shape[0] == 6
    assert r == numpy.sqrt(3)   # is this dumb to test?


def test_face_merge_dodecahedron(regular_dodecahedron_verts):
    origins, normals, r = fresnel.util.convex_polyhedron_from_vertices(regular_dodecahedron_verts)
    assert origins.shape[0] == 12
    assert normals.shape[0] == 12
    assert r == numpy.sqrt(3)


if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene_eight_polyhedra = conftest.scene_eight_polyhedra(device)
    test_render(scene_eight_polyhedra, generate=True)

    scene_eight_polyhedra = conftest.scene_eight_polyhedra(device)
    test_outline(scene_eight_polyhedra, generate=True)

    scene_eight_polyhedra = conftest.scene_eight_polyhedra(device)
    test_face_color(scene_eight_polyhedra, generate=True)
