import pytest
import fresnel
import math
import PIL
import numpy
import os
import pathlib


dir_path = pathlib.Path(os.path.realpath(__file__)).parent

devices = []
if 'cpu' in fresnel.Device.available_modes:
    devices.append(('cpu', None))
    devices.append(('cpu', 1))

if 'gpu' in fresnel.Device.available_modes:
    devices.append(('gpu', 1))


def test_lights():
    lights = []
    phi1 = 1 * 45 * math.pi / 180
    theta1 = (90 - 20) * math.pi / 180
    lights.append(
        fresnel.light.Light(direction=(math.sin(theta1) * math.sin(phi1),
                                       math.cos(theta1),
                                       math.sin(theta1) * math.cos(phi1)),
                            color=(0.75, 0.75, 0.75),
                            theta=math.pi / 8))
    phi1 = -1 * 45 * math.pi / 180
    theta1 = (90) * math.pi / 180
    lights.append(
        fresnel.light.Light(direction=(math.sin(theta1) * math.sin(phi1),
                                       math.cos(theta1),
                                       math.sin(theta1) * math.cos(phi1)),
                            color=(0.1, 0.1, 0.1),
                            theta=math.pi / 2))
    return lights


def device(request):
    mode = request.param[0]
    limit = request.param[1]

    dev = fresnel.Device(mode=mode, n=limit)

    return dev


def device_name(device):
    name = device[0]
    if device[1] is None:
        name += '(all)'
    else:
        name += '(' + str(device[1]) + ')'
    return name


@pytest.fixture(scope='session',
                params=devices,
                ids=device_name)
def device_(request):
    return device(request)


def scene_hex_sphere(device):
    scene = fresnel.Scene(device, lights=test_lights())

    position = []
    for i in range(6):
        position.append([2 * math.cos(i * 2 * math.pi / 6),
                         2 * math.sin(i * 2 * math.pi / 6), 0])

    geometry = fresnel.geometry.Sphere(scene, position=position, radius=1.0)
    geometry.material = fresnel.material.Material(
        solid=0.0, color=fresnel.color.linear([1, 0.874, 0.169]))
    geometry.outline_width = 0.12

    scene.camera = fresnel.camera.Orthographic(
        position=(0, 0, 10), look_at=(0, 0, 0), up=(0, 1, 0), height=6)

    return scene


@pytest.fixture(scope='function')
def scene_hex_sphere_(device_):
    return scene_hex_sphere(device_)


def assert_image_approx_equal(a, ref_file, tolerance=1.0):
    im = PIL.Image.open(ref_file)
    im_arr = numpy.frombuffer(im.tobytes(), dtype=numpy.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 4))

    # intelligently compare images first, ensure that they share a large
    # fraction of non-background pixels (this assumes that all image compare
    # tests use a background alpha = 0)
    a_selection = a[:, :, 3] > 0
    ref_selection = im_arr[:, :, 3] > 0
    assert numpy.sum(a_selection) > 3 / 4 * numpy.sum(ref_selection)

    # Now, compute the sum of the image difference squared, but only over those
    # pixels that are non-background in both images. This prevents a full 255
    # difference from showing up in a pixel that is present in one image but not
    # the other due to a round off error
    selection = a_selection * ref_selection
    a_float = numpy.array(a, dtype=numpy.float32)
    im_float = numpy.array(im_arr, dtype=numpy.float32)
    diff = numpy.array((a_float - im_float))
    msd = numpy.mean(diff[selection]**2)

    assert msd < tolerance


def _validate_or_generate_image(buf, name, generate, tolerance):
    if generate:
        PIL.Image.fromarray(buf[:],
                            mode='RGBA').save(open(f'output/{name}.png', 'wb'),
                                              'png')
    else:
        assert_image_approx_equal(buf[:],
                                  dir_path / 'reference' / f'{name}.png',
                                  tolerance)


def check_preview_render(scene,
                         name,
                         w=150,
                         h=100,
                         tolerance=2,
                         anti_alias=False,
                         generate=False):
    buf_proxy = fresnel.preview(scene, w=w, h=h, anti_alias=anti_alias)

    _validate_or_generate_image(buf_proxy, name, generate, tolerance)


def check_pathtrace_render(scene,
                           name,
                           w=150,
                           h=100,
                           tolerance=2,
                           samples=64 * 40,
                           generate=False):
    buf_proxy = fresnel.pathtrace(scene, w=w, h=h, samples=samples)

    _validate_or_generate_image(buf_proxy, name, generate, tolerance)
