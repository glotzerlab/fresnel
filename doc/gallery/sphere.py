import fresnel, numpy, PIL

data = numpy.load('spheres.npz')

scene = fresnel.Scene()
scene.lights = fresnel.light.cloudy()

geometry = fresnel.geometry.Sphere(
    scene,
    position = data['position'],
    radius = 0.5,
    outline_width = 0.1)

geometry.material = fresnel.material.Material(
    color = fresnel.color.linear([0.1, 0.8, 0.1]),
    roughness = 0.8,
    specular = 0.2)

out = fresnel.pathtrace(scene, samples=64,
                        light_samples=32,
                        w=580, h=580)
PIL.Image.fromarray(out[:], mode='RGBA').save('sphere.png')
