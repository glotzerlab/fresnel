import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest

def test_render(scene_hex_sphere, generate=False):
    tracer = fresnel.tracer.Path(device=scene_hex_sphere.device, w=100, h=100)
    tracer.seed = 11
    buf = tracer.output[:]
    assert buf.shape == (100,100,4)

    buf_proxy = tracer.sample(scene_hex_sphere, samples=64, light_samples=40)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_tracer_path.test_render.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_tracer_path.test_render.png', tolerance=16)

if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('gpu', 1)))

    scene_hex_sphere = conftest.scene_hex_sphere(device)
    test_render(scene_hex_sphere, generate=True)
