import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest

def test_render(scene_four_spheres, generate=False):
    buf_proxy = fresnel.render(scene_four_spheres, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_sphere.test_render.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_sphere.test_render.png')

def test_radius(scene_four_spheres, generate=False):
    geometry = scene_four_spheres.geometry[0]

    r = numpy.array([0.5, 0.6, 0.8, 1.0], dtype=numpy.float32)
    geometry.radius[:] = r
    numpy.testing.assert_array_equal(r, geometry.radius[:])

    buf_proxy = fresnel.render(scene_four_spheres, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_sphere.test_radius.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_sphere.test_radius.png')

def test_position(scene_four_spheres, generate=False):
    geometry = scene_four_spheres.geometry[0]

    p = numpy.array([[1.5,0,1],
                      [1.5,0,-1],
                      [-1.5,0,1],
                      [-1.5,0,-1]], dtype=numpy.float32)
    geometry.position[:] = p
    numpy.testing.assert_array_equal(p, geometry.position[:])

    buf_proxy = fresnel.render(scene_four_spheres, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_sphere.test_position.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_sphere.test_position.png')

def test_color(scene_four_spheres, generate=False):
    geometry = scene_four_spheres.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    c = numpy.array([fresnel.color.linear([1,1,1]), fresnel.color.linear([0,0,1]), fresnel.color.linear([0,1,0]), fresnel.color.linear([1,0,0])], dtype=numpy.float32)
    geometry.color[:] = c
    numpy.testing.assert_array_equal(c, geometry.color[:])

    buf_proxy = fresnel.render(scene_four_spheres, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_sphere.test_color.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_sphere.test_color.png')

def test_outline(scene_four_spheres, generate=False):
    geometry = scene_four_spheres.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.render(scene_four_spheres, w=150, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_sphere.test_outline.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_sphere.test_outline.png')

if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene_four_spheres = conftest.scene_four_spheres(device)
    test_render(scene_four_spheres, generate=True)

    scene_four_spheres = conftest.scene_four_spheres(device)
    test_radius(scene_four_spheres, generate=True)

    scene_four_spheres = conftest.scene_four_spheres(device)
    test_position(scene_four_spheres, generate=True)

    scene_four_spheres = conftest.scene_four_spheres(device)
    test_color(scene_four_spheres, generate=True)

    scene_four_spheres = conftest.scene_four_spheres(device)
    test_outline(scene_four_spheres, generate=True)
