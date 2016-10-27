# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Geometric primitives.
"""

from . import material

class Geometry:
    R""" Base class for all geometry.

    :py:class:`Geometry` provides operations common to all geometry classes.

    Note:

        You cannot instantiate a Geometry directly. Use one of the sub classes.

    """
    def __init__(self):
        raise RuntimeError("Use a specific geometry class");

    def enable(self):
        R""" Enable the geometry.

        When enabled, the geometry will be present when rendering the scene.
        """

        self._geometry.enable();

    def disable(self):
        R""" Disable the geometry.

        When disabled, the geometry will not be present in the scene. No rays will intersect it.
        """

        self._geometry.disable();

    def remove(self):
        R""" Remove the geometry from the scene.

        After calling remove, the geometry is no longer part of the scene. It cannot be added back into the scene.
        Use :py:meth:`disable` if you want a reversible operation.
        """
        self._geometry.remove();
        self.scene.geometry.remove(self)

    @property
    def material(self):
        return material._material_proxy(self);

    @material.setter
    def material(self, mat):
        self._geometry.setMaterial(mat._get_cpp_material());

class TriangleMesh(Geometry):
    R""" Triangle mesh geometry.

    Define a geometry consisting of a set of triangles.

    Warning:

        This class is temporary, for prototype testing only. It may be removed
    """

    def __init__(self, scene, verts, indices, material=material.Material(solid=1.0, color=(1,0,1))):
        self._geometry = scene.device.module.GeometryTriangleMesh(scene._scene, verts, indices);
        self.material = material;

        self.scene = scene;
        self.scene.geometry.append(self);

class Prism(Geometry):
    R""" Prism geometry.

    Define a set of right convex prism primitives. In this implementation, the bottom polygon face is always in the
    xy plane and each prism may have a different height and rotation angle.

    Warning:

        This class is  a prototype for testing, its API may change drastically.
    """

    def __init__(self, scene, verts, position, angle, height, color, material=material.Material(solid=1.0, color=(1,0,1))):
        self._geometry = scene.device.module.GeometryPrism(scene._scene, verts, position, angle, height, color);
        self.material = material;

        self.scene = scene;
        self.scene.geometry.append(self);

class Sphere(Geometry):
    R""" Sphere geometry.

    Define a set of sphere primitives with positions and radii

    Warning:

        This class is  a prototype for testing, its API may change drastically.
    """

    def __init__(self, scene, position, radii, material=material.Material(solid=1.0, color=(1,0,1))):
        self._geometry = scene.device.module.GeometrySphere(scene._scene, position, radii);
        self.material = material;

        self.scene = scene;
        self.scene.geometry.append(self);

class Sphere(Geometry):
    R""" Sphere geometry.

    Define a set of sphere primitives with positions and radii

    Warning:

        This class is  a prototype for testing, its API may change drastically.
    """

    def __init__(self, scene, position, radii, material=material.Material(solid=1.0, color=(1,0,1))):
        self._geometry = scene.device.module.GeometrySphere(scene._scene, position, radii);
        self.material = material;

        self.scene = scene;
        self.scene.geometry.append(self);
