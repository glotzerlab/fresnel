from __future__ import division

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

@pytest.fixture(scope='function')
def scene_hex_sphere(device):
    scene = fresnel.Scene(device)

    position = []
    for i in range(6):
        position.append([2*math.cos(i*2*math.pi / 6), 2*math.sin(i*2*math.pi / 6), 0])

    geometry = fresnel.geometry.Sphere(scene, position = position, radius=1.0)
    geometry.material = fresnel.material.Material(solid=0.0, color=fresnel.color.linear([1,0.874,0.169]))
    geometry.outline_width = 0.12

    scene.camera = fresnel.camera.Orthographic(position=(0, 0, 10), look_at=(0,0,0), up=(0,1,0), height=6)

    return scene

@pytest.fixture(scope='function')
def scene_four_spheres(device):
    scene = fresnel.Scene(device)

    position = []
    for i in range(6):
        position.append([2*math.cos(i*2*math.pi / 6), 2*math.sin(i*2*math.pi / 6), 0])

    geometry = fresnel.geometry.Sphere(scene,
                                       position = [[1,0,1],
                                                   [1,0,-1],
                                                   [-1,0,1],
                                                   [-1,0,-1]],
                                       radius=1.0,
                                       material = fresnel.material.Material(color=fresnel.color.linear([0.42,0.267,1])),
                                       color = [[1,0,0], [0,1,0], [0,0,1], [1,0,1]]
                                       )

    scene.camera = fresnel.camera.Orthographic(position=(10, 10, 10), look_at=(0,0,0), up=(0,1,0), height=4)
    scene.light_direction = (4,3,0)  # Note: this API is temporary

    return scene

def assert_image_approx_equal(a, ref_file):
    im = PIL.Image.open(ref_file)
    im_arr = numpy.fromstring(im.tobytes(), dtype=numpy.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 4))

    # intelligently compare images
    # first, ensure that they share a large fraction of non-background pixels (this assumes that all image compare
    # tests use a background alpha = 0)
    a_selection = a[:,:,3] > 0;
    ref_selection = a[:,:,3] > 0;
    assert numpy.sum(a_selection) > 3/4 * numpy.sum(ref_selection)

    # Now, compute the sum of the image difference squared, but only over those pixels that are
    # non-background in both images. This prevents a full 255 difference from showing up in
    # a pixel that is present in one image but not the other due to a round off error
    selection = a_selection * ref_selection
    a_float = numpy.array(a, dtype=numpy.float32)
    im_float = numpy.array(im_arr, dtype=numpy.float32)
    diff = numpy.array((a_float - im_float))
    msd = numpy.mean(diff[selection]**2)

    assert msd < 1.0

