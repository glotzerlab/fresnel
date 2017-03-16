import fresnel

def test_camera_fit_front(device):
    scene = fresnel.Scene()

    geom1 = fresnel.geometry.Sphere(scene, position = [[-9, -2, 0], [-5, -1, 0], [4, 0, 0], [2, 1, 0]], radius=1.0)

    cam = fresnel.camera.fit(scene, view='front', margin=0)
    assert cam.position[0] == -2.5
    assert cam.position[1] == -0.5
    assert cam.look_at == (-2.5,-0.5,0)
    assert cam.height == 5

def test_camera_fit_isometric(device):
    scene = fresnel.Scene()

    geom1 = fresnel.geometry.Sphere(scene, position = [[-9, -2, 0], [-5, -1, 0], [4, 0, 0], [2, 1, 0]], radius=1.0)

    cam = fresnel.camera.fit(scene, view='isometric', margin=0)
    # isometric cameras do not have a simple testable format, just test that the API works

def test_scene_auto(device):
    scene = fresnel.Scene()

    geom1 = fresnel.geometry.Sphere(scene, position = [[-9, -2, 0], [-5, -1, 0], [4, 0, 0], [2, 1, 0]], radius=1.0)
    assert scene.camera == 'auto';

    fresnel.render(scene)

def test_orthographic_attributes():
    cam = fresnel.camera.orthographic(position=(1, 0, 10), look_at=(1,0,0), up=(0,1,0), height=6)
    assert cam.position == (1,0,10)
    assert cam.look_at == (1,0,0)
    assert cam.up == (0,1,0)
    assert cam.height == 6

    cam2 = eval(repr(cam))
    assert cam.position == (1,0,10)
    assert cam.look_at == (1,0,0)
    assert cam.up == (0,1,0)
    assert cam.height == 6

    cam.position = (1,3,8)
    cam.look_at = (20,5,18)
    cam.up = (1,2,3)
    cam.height = 112

    assert cam.position == (1,3,8)
    assert cam.look_at == (20,5,18)
    assert cam.up == (1,2,3)
    assert cam.height == 112
