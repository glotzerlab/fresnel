import fresnel

def test_lightlist():
    l = fresnel.light.LightList()
    assert(len(l) == 0)
    l.append(direction=(1,2,3), color=(0.5, 0.125, 0.75))
    assert(len(l) == 1)
    l.append(direction=(5,10,2), color=(10, 18, 6))
    assert(len(l) == 2)
    l.append(direction=(4,8,16), color=(100, 1200, 7))
    assert(len(l) == 3)
    l.append(direction=(5,1,2), color=(16, 32, 64))
    assert(len(l) == 4)

    assert(l[0].direction == (1,2,3))
    assert(l[0].color == (0.5, 0.125, 0.75))

    assert(l[1].direction == (5,10,2))
    assert(l[1].color == (10, 18, 6))

    assert(l[2].direction == (4,8,16))
    assert(l[2].color == (100, 1200, 7))

    assert(l[3].direction == (5,1,2))
    assert(l[3].color == (16, 32, 64))

    l[0].direction = (10,1,18)
    l[0].color = (5,10,20)
    assert(l[0].direction == (10,1,18))
    assert(l[0].color == (5,10,20))

    l.clear()
    assert(len(l) == 0)
