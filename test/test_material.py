import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import os

def test_set_material(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material = fresnel.material.Material(solid=0.0, color=fresnel.color.linear([1,0,0]), primitive_color_mix=0.0)
    assert geometry.material.solid == 0.0
    assert geometry.material.color == tuple(fresnel.color.linear([1,0,0]))
    assert geometry.material.primitive_color_mix == 0.0

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_set_material.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_set_material.png'))

def test_solid(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material.solid = 1.0
    assert geometry.material.solid == 1.0

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_solid.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_solid.png'))

def test_color(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material.color = fresnel.color.linear([0,0,1])
    assert geometry.material.color == tuple(fresnel.color.linear([0,0,1]))

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_color.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_color.png'))

def test_specular(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material.specular = 1.0
    assert geometry.material.specular == 1.0

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_specular.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_specular.png'))

def test_roughness(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material.roughness = 1.0
    assert geometry.material.roughness == 1.0

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_roughness.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_roughness.png'))

def test_metal(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material.metal = 1.0
    assert geometry.material.metal == 1.0

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_metal.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_metal.png'))

def test_primitive_color_mix(scene_hex_sphere_, pytestconfig, generate=False):
    geometry = scene_hex_sphere_.geometry[0]
    geometry.material = fresnel.material.Material(solid=1.0, color=fresnel.color.linear([1,0,0]), primitive_color_mix=1.0)

    geometry.color[0] = fresnel.color.linear([1,0,0])
    geometry.color[1] = fresnel.color.linear([0,1,0])
    geometry.color[2] = fresnel.color.linear([0,0,1])
    geometry.color[3] = fresnel.color.linear([1,0,1])
    geometry.color[4] = fresnel.color.linear([0,1,1])
    geometry.color[5] = fresnel.color.linear([0,0,0])

    buf_proxy = fresnel.preview(scene_hex_sphere_, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_material.test_primitive_color_mix.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], os.path.join(str(pytestconfig.rootdir),'test/reference/test_material.test_primitive_color_mix.png'))

if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = conftest.scene_hex_sphere(device)
    test_set_material(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_solid(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_color(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_primitive_color_mix(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_specular(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_roughness(scene, None, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_metal(scene, None, generate=True)
