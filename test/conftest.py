from __future__ import division

import pytest
import fresnel
import itertools
import math
import PIL
import numpy

devices = []
if 'cpu' in fresnel.Device.available_modes:
    devices.append( ('cpu', None) );
    devices.append( ('cpu', 1) );

if 'gpu' in fresnel.Device.available_modes:
    devices.append( ('gpu', 1) );

def test_lights():
    lights = [];
    phi1 = 1*45*math.pi/180;
    theta1 = (90-20)*math.pi/180;
    lights.append(fresnel.light.Light(direction=(math.sin(theta1)*math.sin(phi1),
                                                 math.cos(theta1),
                                                 math.sin(theta1)*math.cos(phi1)),
                                      color=(0.75,0.75,0.75),
                                      theta=math.pi/8));
    phi1 = -1*45*math.pi/180;
    theta1 = (90)*math.pi/180;
    lights.append(fresnel.light.Light(direction=(math.sin(theta1)*math.sin(phi1),
                                                 math.cos(theta1),
                                                 math.sin(theta1)*math.cos(phi1)),
                                      color=(0.1,0.1,0.1),
                                      theta=math.pi/2));
    return lights;

@pytest.fixture(scope='session',
                params=devices)
def device(request):
    mode = request.param[0]
    limit = request.param[1]

    dev = fresnel.Device(mode=mode, n=limit)

    return dev

@pytest.fixture(scope='function')
def scene_hex_sphere(device):
    scene = fresnel.Scene(device, lights = test_lights())

    position = []
    for i in range(6):
        position.append([2*math.cos(i*2*math.pi / 6), 2*math.sin(i*2*math.pi / 6), 0])

    geometry = fresnel.geometry.Sphere(scene, position = position, radius=1.0)
    geometry.material = fresnel.material.Material(solid=0.0, color=fresnel.color.linear([1,0.874,0.169]))
    geometry.outline_width = 0.12

    scene.camera = fresnel.camera.orthographic(position=(0, 0, 10), look_at=(0,0,0), up=(0,1,0), height=6)

    return scene

@pytest.fixture(scope='function')
def scene_four_spheres(device):
    scene = fresnel.Scene(device, lights = test_lights())

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

    scene.camera = fresnel.camera.orthographic(position=(10, 10, 10), look_at=(0,0,0), up=(0,1,0), height=4)

    return scene

@pytest.fixture(scope='function')
def scene_eight_polyhedra(device):
    scene = fresnel.Scene(device, lights = test_lights())

    # place eight polyhedra
    position = []
    for k in range(2):
        for i in range(2):
            for j in range(2):
                position.append([2.5*i, 2.5*j, 2.5*k])


    # create the polyhedron faces
    origins=[];
    normals=[];
    colors=[];

    for v in [-1, 1]:
        origins.append([v, 0, 0])
        normals.append([v, 0, 0])
        origins.append([0, v, 0])
        normals.append([0, v, 0])
        origins.append([0, 0, v])
        normals.append([0, 0, v])
        colors.append([178/255,223/255,138/255])
        colors.append([178/255,223/255,138/255])
        colors.append([178/255,223/255,138/255])

    for x in [-1,1]:
        for y in [-1,1]:
            for z in [-1,1]:
                normals.append([x,y,z])
                origins.append([x*0.75, y*0.75, z*0.75])
                colors.append([166/255,206/255,227/255])

    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                 origins=origins,
                                                 normals=normals,
                                                 face_colors = fresnel.color.linear(colors),
                                                 r=math.sqrt(3),
                                                 position=position)

    geometry.material = fresnel.material.Material(color=fresnel.color.linear([1.0,0, 0]),
                                                 roughness=0.8,
                                                 specular=0.5,
                                                 primitive_color_mix = 0.0)
    geometry.orientation[:] = [1,0,0,0]

    scene.camera = fresnel.camera.orthographic(position=(20, 20, 20), look_at=(0,0,0), up=(0,1,0), height=7)

    return scene


@pytest.fixture(scope='function')
def cube_verts():
    pms = [+1, -1]
    return numpy.array([x for x in itertools.product(pms, repeat=3)])


@pytest.fixture(scope='function')
def regular_dodecahedron_verts():
    phi = (1 + numpy.sqrt(5)) / 2
    vertices = numpy.array([[1, 1, 1],
                        [1, 1, -1],
                        [1, -1, 1],
                       [-1, 1, 1],
                       [1, -1, -1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [-1, -1, -1],
                       [0, phi, 1/phi],
                       [0, -phi, 1/phi],
                       [0, phi, -1/phi],
                       [0, -phi, -1/phi],
                       [1/phi, 0, phi],
                       [1/phi, 0, -phi],
                       [-1/phi, 0, phi],
                       [-1/phi, 0, -phi],
                       [phi, 1/phi, 0],
                       [phi, -1/phi, 0],
                       [-phi, 1/phi, 0],
                       [-phi, -1/phi, 0]])
    return vertices


def assert_image_approx_equal(a, ref_file, tolerance=1.0):
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

    assert msd < tolerance

