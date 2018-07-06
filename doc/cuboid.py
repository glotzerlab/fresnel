import fresnel, numpy, math, PIL

data = numpy.load('cuboids.npz')

scene = fresnel.Scene()
scene.lights = fresnel.light.lightbox()

geometry = fresnel.geometry.ConvexPolyhedron(
    scene,
    origins = [[-data['width'][0],0,0], [data['width'][0],0,0], [0, -data['width'][1], 0],
               [0, data['width'][1], 0], [0, 0, -data['width'][2]], [0, 0, data['width'][2]]],
    normals = [[-1,0,0], [1,0,0], [0, -1, 0],
               [0, 1, 0],[0, 0, -1], [0, 0, 1]],
    r = math.sqrt(data['width'][0]**2 + data['width'][1]**2 + data['width'][2]**2),
    position = data['position'],
    orientation = data['orientation'],
    outline_width = 0.015)
geometry.material = fresnel.material.Material(
    color = fresnel.color.linear([0.1, 0.1, 0.6]),
    roughness = 0.1,
    specular = 1)
geometry.outline_material = fresnel.material.Material(
    color = (0.95,0.93,0.88),
    roughness = 0.1,
    metal = 1.0)

scene.camera = fresnel.camera.fit(scene, view='front')

out = fresnel.pathtrace(scene, samples=64, light_samples=32, w=500, h=500)
PIL.Image.fromarray(out[:], mode='RGBA').save('cuboid.png')
