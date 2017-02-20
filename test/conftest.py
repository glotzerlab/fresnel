import pytest
import fresnel
import math
import PIL
import numpy

@pytest.fixture(scope='session',
                params=[('cpu', None), ('cpu', 1), ('gpu', None)])
def device(request):
    mode = request.param[0]
    limit = request.param[1]

    if mode == 'cpu':
        dev = fresnel.Device(mode=mode, limit=limit)
    else:
        dev = fresnel.Device(mode=mode)

    return dev

@pytest.fixture(scope='session')
def scene_hex_sphere(device):
    scene = fresnel.Scene(device)

    position = []
    for i in range(6):
        position.append([2*math.cos(i*2*math.pi / 6), 2*math.sin(i*2*math.pi / 6), 0])

    geometry = fresnel.geometry.Sphere(scene, position = position, radius=1.0)
    geometry.material = fresnel.material.Material(solid=0.0, color=fresnel.color.linear([1,0.874,0.169]))
    geometry.outline_width = 0.12

    return scene

def assert_image_approx_equal(a, ref_file):
    im = PIL.Image.open(ref_file)
    im_arr = numpy.fromstring(im.tobytes(), dtype=numpy.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 4))

    diff = numpy.array((a - im_arr).flatten(), dtype=numpy.float32)
    msd = numpy.mean(diff**2)

    assert msd < 1.0

