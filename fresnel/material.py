# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Materials describe the way light interacts with surfaces.
"""

from fresnel import _common

class Material:
    R"""Define material properties.

    Args:

        solid (float): Set to 1 to pass through a solid color, regardless of the light and view angle.
        color (tuple): 3-tuple, list or other iterable that specifies the RGB color of the material.
        borrow: For internal use.

    TODO: Document SRGB and linear color spaces.
    """

    def __init__(self, solid=0, color=(0,0,0), borrow=None):
        self.borrowed = borrow;

        if borrow is None:
            self._material = _common.Material();

            self.solid = solid;
            self.color = color;
        else:
            # when borrow is not none, simply borrow the C++ material instance and refer to it
            # do not set new material parameters
            self._material = borrow;

    @property
    def solid(self):
        return self._material.solid;

    @solid.setter
    def solid(self, value):
        if self.borrowed:
            raise AttributeError("Individual material properties of geometries are read only.")
        self._material.solid = float(value);

    @property
    def color(self):
        return (self._material.color.r, self._material.color.g, self._material.color.b)

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3");
        if self.borrowed:
            raise AttributeError("Individual material properties of geometries are read only.")

        self._material.color = _common.RGBf(value[0], value[1], value[2]);

