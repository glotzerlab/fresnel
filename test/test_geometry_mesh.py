import fresnel
import numpy
from collections import namedtuple
import PIL
import conftest
import pytest
import sys

@pytest.fixture(scope='function')
def scene_one_triangle(device):
    scene = fresnel.Scene(device, lights = conftest.test_lights())

    geometry = fresnel.geometry.Mesh(scene,vertices=[[-1,-1,0],[1,-1,0],[-1,1,0]],N=1)
    geometry.color[:] = [[1,0,0], [0,1,0], [0,0,1]]

    geometry.material = fresnel.material.Material(color=fresnel.color.linear([1.0,0, 0]),
                                                 roughness=0.8,
                                                 specular=0.5,
                                                 primitive_color_mix = 0.0,
                                                 solid=1)

    geometry.outline_material = fresnel.material.Material(color=fresnel.color.linear([0, 1.0, 0]),
                                                 roughness=0.8,
                                                 specular=0.5,
                                                 primitive_color_mix = 0.0,
                                                 solid=1)

    geometry.orientation[:] = [1,0,0,0]

    scene.camera = fresnel.camera.orthographic(position=(0, 0, 20), look_at=(0,0,0), up=(0,1,0), height=2.1)

    return scene

@pytest.fixture(scope='function')
def scene_tetrahedra(device):
    scene = fresnel.Scene(device, lights = conftest.test_lights())

    verts = [(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)]
    triangles = [verts[0], verts[1], verts[2],
                 verts[2], verts[1], verts[3],
                 verts[2], verts[3], verts[0],
                 verts[1], verts[0], verts[3]]

    geometry = fresnel.geometry.Mesh(scene,vertices=triangles,N=4)
    geometry.color[0:2,:] = [0.9,0,0];
    geometry.color[3:5,:] = [0,0.9,0];
    geometry.color[6:8,:] = [0,0,0.9];
    geometry.color[9:11,:] = [0.9,0,0.9];

    geometry.material = fresnel.material.Material(color=fresnel.color.linear([1.0,0, 0]),
                                                 roughness=0.8,
                                                 specular=0.5,
                                                 primitive_color_mix = 1.0,
                                                 solid=0)

    geometry.position[:] = [[-2, -2, 0],
                            [2, -2 ,0],
                            [2, 2, 0],
                            [-2, 2, 0]];

    geometry.orientation[:] = [[ 0.03723867,  0.38927173, -0.73216521, -0.55768711],
        [-0.32661186,  0.43644863, -0.09899935,  0.83248808],
        [ 0.25624845,  0.32632096, -0.11995704, -0.9019211 ],
        [-0.78025512, -0.12102377,  0.24947819,  0.56063877]];

    scene.camera = fresnel.camera.orthographic(position=(0, 0, -20), look_at=(0,0,0), up=(0,1,0), height=7.5)

    return scene

def test_render(scene_one_triangle, generate=False):
    buf_proxy = fresnel.preview(scene_one_triangle, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_mesh.test_render.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_mesh.test_render.png')


def test_outline(scene_one_triangle, generate=False):
    geometry = scene_one_triangle.geometry[0]
    geometry.outline_width = 0.1

    buf_proxy = fresnel.preview(scene_one_triangle, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_mesh.test_outline.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_mesh.test_outline.png')

def test_color_interp(scene_one_triangle, generate=False):
    geometry = scene_one_triangle.geometry[0]
    geometry.material.primitive_color_mix = 1.0

    buf_proxy = fresnel.preview(scene_one_triangle, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_mesh.test_color_interp.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_mesh.test_color_interp.png')

def test_multiple(scene_tetrahedra, generate=False):
    buf_proxy = fresnel.preview(scene_tetrahedra, w=100, h=100)

    if generate:
        PIL.Image.fromarray(buf_proxy[:], mode='RGBA').save(open('output/test_geometry_mesh.test_multiple.png', 'wb'), 'png');
    else:
        conftest.assert_image_approx_equal(buf_proxy[:], 'reference/test_geometry_mesh.test_multiple.png')

if __name__ == '__main__':
    struct = namedtuple("struct", "param")
    device = conftest.device(struct(('cpu', None)))

    scene = scene_one_triangle(device)
    test_render(scene, generate=True)

    scene = scene_one_triangle(device)
    test_outline(scene, generate=True)

    scene = scene_one_triangle(device)
    test_color_interp(scene, generate=True)

    scene = scene_tetrahedra(device)
    test_multiple(scene, generate=True)
