import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import os
import pathlib

dir_path = pathlib.Path(os.path.realpath(__file__)).parent

def test_render(scene_hex_sphere_, generate=False):
    tracer = fresnel.tracer.Preview(device=scene_hex_sphere_.device, w=100, h=100, anti_alias=False)
    buf = tracer.output[:]
    assert buf.shape == (100,100,4)

    buf_proxy = tracer.render(scene_hex_sphere_)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_tracer_direct.test_render.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], dir_path / 'reference' / 'test_tracer_direct.test_render.png')

def test_render_aa(scene_hex_sphere_, generate=False):
    tracer = fresnel.tracer.Preview(device=scene_hex_sphere_.device, w=100, h=100, anti_alias=True)
    buf = tracer.output[:]
    assert buf.shape == (100,100,4)

    buf_proxy = tracer.render(scene_hex_sphere_)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_tracer_direct.test_render_aa.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], dir_path / 'reference' / 'test_tracer_direct.test_render_aa.png')

def test_resize(scene_hex_sphere_, generate=False):
    tracer = fresnel.tracer.Preview(device=scene_hex_sphere_.device, w=100, h=100, anti_alias=False)
    buf = tracer.output[:]
    assert buf.shape == (100,100,4)

    tracer.resize(w=200, h=300)
    buf = tracer.output[:]
    assert buf.shape == (300,200,4)

if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = conftest.scene_hex_sphere(device)
    test_render(scene, generate=True)

    scene = conftest.scene_hex_sphere(device)
    test_render_aa(scene, generate=True)
