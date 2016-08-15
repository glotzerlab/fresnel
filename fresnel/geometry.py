# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Geometric primitives.
"""

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

class TriangleMesh(Geometry):
    R""" Triangle mesh geometry.

    Define a geometry consisting of a set of triangles.

    Warning:

        This class is temporary, for prototype testing only. It may be removed
    """

    def __init__(self, scene, verts, indices):
        self._geometry = scene.device.module.GeometryTriangleMesh(scene._scene, verts, indices);
