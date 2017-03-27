import fresnel
import pytest

def test_lightlist(device):
    scene = fresnel.Scene(device, lights=[])
    l = scene.lights
    assert(len(l) == 0)
    l.append(fresnel.light.Light(direction=(1,2,3), color=(0.5, 0.125, 0.75)))
    assert(len(l) == 1)
    l.append(fresnel.light.Light(direction=(5,10,2), color=(10, 18, 6)))
    assert(len(l) == 2)
    l.append(fresnel.light.Light(direction=(4,8,16), color=(100, 1200, 7)))
    assert(len(l) == 3)
    l.append(fresnel.light.Light(direction=(5,1,2), color=(16, 32, 64)))
    assert(len(l) == 4)

    with pytest.raises(Exception):
        l.append(fresnel.light.Light(direction=(5,1,2), color=(16, 32, 64)))

    assert(l[0].direction == (1,2,3))
    assert(l[0].color == (0.5, 0.125, 0.75))

    assert(l[1].direction == (5,10,2))
    assert(l[1].color == (10, 18, 6))

    assert(l[2].direction == (4,8,16))
    assert(l[2].color == (100, 1200, 7))

    assert(l[3].direction == (5,1,2))
    assert(l[3].color == (16, 32, 64))

    with pytest.raises(Exception):
        v = l[4].direction;

    l[0].direction = (10,1,18)
    l[0].color = (5,10,20)
    assert(l[0].direction == (10,1,18))
    assert(l[0].color == (5,10,20))

    for light in l:
        d = light.direction;
        light.direction = (-d[0], -d[1], -d[2])

    assert(l[0].direction == (-10,-1,-18))
    assert(l[1].direction == (-5,-10,-2))
    assert(l[2].direction == (-4,-8,-16))
    assert(l[3].direction == (-5,-1,-2))

    l.clear()
    assert(len(l) == 0)

    scene.lights = [fresnel.light.Light(direction=(1,2,3)), fresnel.light.Light(direction=(10,20,30), color=(1,0,0))];
    assert(l[0].direction == (1,2,3))
    assert(l[0].color == (1,1,1))
    assert(l[1].direction == (10,20,30))
    assert(l[1].color == (1,0,0))

    with pytest.raises(Exception):
        scene.lights = [fresnel.light.Light(direction=(1,2,3)),
                        fresnel.light.Light(direction=(10,20,30), color=(1,0,0)),
                        fresnel.light.Light(direction=(1,2,3)),
                        fresnel.light.Light(direction=(1,2,3)),
                        fresnel.light.Light(direction=(1,2,3))];
