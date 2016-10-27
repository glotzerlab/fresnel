# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Geometric primitives.
"""

from . import material
from . import util

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

class Prism(Geometry):
    R""" Prism geometry.

    Define a set of right convex prism primitives. The bottom polygon face is always in the xy plane. Each prism may
    have a different height and rotation angle.

    Args:
        scene (:py:class:`fresnel.Scene`): Add the geometry to this scene
        vertices: Mx2 numpy array or data type convertible by numpy to a Nx2 array of floats. Specify the vertices of
          the polygon in a counter clockwise winding direction.
        position: Nx2 numpy array or data type convertible by numpy to a Nx2 array of floats. Specifies the positions
          of the prisms. *optional*
        height: N length numpy array or data type convertible by numpy to a N length array of floats. Specifies the
          height of each prism in the z direction. *optional*
        angle: N length numpy array or data type convertible by numpy to a N length array of floats. Specifies the
          rotation angle of each prism (in radians). *optional*
        color: Nx3 length numpy array or data type convertible by numpy to a Nx3 length array of floats. Specifies the (r,g,b)
          color of each particle. *optional*
        N (int): Number of spheres in the geometry. If ``None``, determine ``N`` from ``position``.

    .. hint::
        Avoid costly memory allocations and type conversions by specifying geometry attributes in the appropriate
        numpy array type.

    Attributes:
        position (:py:class:`fresnel.util.array`): Read or modify the positions of the prisms.
        height (:py:class:`fresnel.util.array`): Read or modify the heights of the prisms.
        angle (:py:class:`fresnel.util.array`): Read or modify the angles of the prisms.
        color (:py:class:`fresnel.util.array`): Read or modify the color of the prisms.

    """

    def __init__(self, scene, vertices, position=None, angle=None, height=None, color=None, N=None, material=material.Material(solid=1.0, color=(1,0,1))):
        if N is None:
            N = len(position);

        self._geometry = scene.device.module.GeometryPrism(scene._scene, vertices, N);
        self.material = material;

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
        position: Nx3 numpy array or data type convertible by numpy to a Nx3 array of floats. Specifies the positions of the
          spheres. *optional*
        radius: N length numpy array or data type convertible by numpy to a N length array of floats. Specifies the radii of the
          spheres. *optional*
        color: Nx3 length numpy array or data type convertible by numpy to a Nx3 length array of floats. Specifies the (r,g,b)
          color of each particle. *optional*
        N (int): Number of spheres in the geometry. If ``None``, determine ``N`` from ``position``.

    .. hint::
        Avoid costly memory allocations and type conversions by specifying geometry attributes in the appropriate
        numpy array type.

    Attributes:
        position (:py:class:`fresnel.util.array`): Read or modify the positions of the spheres.
        radius (:py:class:`fresnel.util.array`): Read or modify the radii of the spheres.
        color (:py:class:`fresnel.util.array`): Read or modify the color of the spheres.
    """

    def __init__(self, scene, position=None, radius=None, color=None, N=None, material=material.Material(solid=1.0, color=(1,0,1))):
        if N is None:
            N = len(position);

        self._geometry = scene.device.module.GeometrySphere(scene._scene, N);
        self.material = material;

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
