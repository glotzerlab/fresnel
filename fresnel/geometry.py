# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Geometric primitives.
"""

from . import material
from . import util

class Geometry:
    R""" Base class for all geometry.

    :py:class:`Geometry` provides operations common to all geometry classes.

    Attributes:
        material (:py:class:`fresnel.material.Material`): Read, set, or modify the geometry's material.
        outline_material (:py:class:`fresnel.material.Material`): Read, set, or modify the geometry's outline material.
        outline_width (:any:`float`): The geometry's outline width, in distance units in the scene's coordinate system.

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

    @property
    def outline_material(self):
        return material._outline_material_proxy(self);

    @outline_material.setter
    def outline_material(self, mat):
        self._geometry.setOutlineMaterial(mat._get_cpp_material());

    @property
    def outline_width(self):
        return self._geometry.getOutlineWidth();

    @outline_width.setter
    def outline_width(self, width):
        self._geometry.setOutlineWidth(width);

class Prism(Geometry):
    R""" Prism geometry.

    Define a set of right convex prism primitives. The bottom polygon face is always in the xy plane. Each prism may
    have a different height and rotation angle.

    Args:
        scene (:py:class:`fresnel.Scene`): Add the geometry to this scene
        vertices: The vertices of the polygon in a counter clockwise winding direction.
          **Type:** anything convertible by numpy to a Nx2 array of floats.
        position: Positions of the prisms, *optional*.
          **Type:** anything convertible by numpy to a Nx3 array of floats.
        height: Height of each prism in the z direction, *optional*.
          **Type:** anything convertible by numpy to a N length array of floats.
        angle: Rotation angle of each prism (in radians), *optional*.
          **Type:** anything convertible by numpy to a N length array of floats.
        color: (r,g,b) color of each particle, *optional*.
          **Type:** anything convertible by numpy to a Nx3 array of floats.
        N (int): Number of spheres in the geometry. If ``None``, determine ``N`` from ``position``.

    Note:
        The constructor arguments ``position``, ``height``, ``angle``, and ``color`` are optional, and just short-hand
        for assigning the attribute after construction.

    .. hint::
        Avoid costly memory allocations and type conversions by specifying primitive properties in the appropriate
        numpy array type.

    Attributes:
        position (:py:class:`fresnel.util.array`): Read or modify the positions of the prisms.
        height (:py:class:`fresnel.util.array`): Read or modify the heights of the prisms.
        angle (:py:class:`fresnel.util.array`): Read or modify the angles of the prisms.
        color (:py:class:`fresnel.util.array`): Read or modify the color of the prisms.

    """

    def __init__(self,
                 scene,
                 vertices,
                 position=None,
                 angle=None,
                 height=None,
                 color=None,
                 N=None,
                 material=material.Material(solid=1.0, color=(1,0,1)),
                 outline_material=material.Material(solid=1.0, color=(0,0,0)),
                 outline_width=0.0):
        if N is None:
            N = len(position);

        self._geometry = scene.device.module.GeometryPrism(scene._scene, vertices, N);
        self.material = material;
        self.outline_material = outline_material;
        self.outline_width = outline_width;

        if position is not None:
            self.position[:] = position;

        if height is not None:
            self.height[:] = height;

        if angle is not None:
            self.angle[:] = angle;

        if color is not None:
            self.color[:] = color;

        self.scene = scene;
        self.scene.geometry.append(self);

    @property
    def position(self):
        return util.array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def height(self):
        return util.array(self._geometry.getHeightBuffer(), geom=self)

    @property
    def angle(self):
        return util.array(self._geometry.getAngleBuffer(), geom=self)

    @property
    def color(self):
        return util.array(self._geometry.getColorBuffer(), geom=self)


class Sphere(Geometry):
    R""" Sphere geometry.

    Define a set of sphere primitives with positions, radii, and individual colors.

    Args:
        scene (:py:class:`fresnel.Scene`): Add the geometry to this scene
        position: Positions of the spheres, *optional*.
          **Type:** anything convertible by numpy to a Nx3 array of floats.
        radius: Radius of each sphere, *optional*.
          **Type:** anything convertible by numpy to a N length array of floats.
        color: (r,g,b) color of each particle, *optional*.
          **Type:** anything convertible by numpy to a Nx3 array of floats.
        N (int): Number of spheres in the geometry. If ``None``, determine ``N`` from ``position``.

    Note:
        The constructor arguments ``position``, ``radius``, and ``color`` are optional, and just short-hand
        for assigning the properties after construction.

    .. hint::
        Avoid costly memory allocations and type conversions by specifying primitive properties in the appropriate
        numpy array type.

    Attributes:
        position (:py:class:`fresnel.util.array`): Read or modify the positions of the spheres.
        radius (:py:class:`fresnel.util.array`): Read or modify the radii of the spheres.
        color (:py:class:`fresnel.util.array`): Read or modify the color of the spheres.
    """

    def __init__(self,
                 scene,
                 position=None,
                 radius=None,
                 color=None,
                 N=None,
                 material=material.Material(solid=1.0, color=(1,0,1)),
                 outline_material=material.Material(solid=1.0, color=(0,0,0)),
                 outline_width=0.0):
        if N is None:
            N = len(position);

        self._geometry = scene.device.module.GeometrySphere(scene._scene, N);
        self.material = material;
        self.outline_material = outline_material;
        self.outline_width = outline_width;

        if position is not None:
            self.position[:] = position;

        if radius is not None:
            self.radius[:] = radius;

        if color is not None:
            self.color[:] = color;

        self.scene = scene;
        self.scene.geometry.append(self);

    @property
    def position(self):
        return util.array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def radius(self):
        return util.array(self._geometry.getRadiusBuffer(), geom=self)

    @property
    def color(self):
        return util.array(self._geometry.getColorBuffer(), geom=self)
