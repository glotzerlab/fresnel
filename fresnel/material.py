# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Materials describe the way light interacts with surfaces.
"""

from fresnel import _common

class Material:
    R"""Define material properties.

    Args:

        solid (float): Set to 1 to pass through a solid color, regardless of the light and view angle.
        color (tuple): The RGB color of the material as a 3-tuple, list or other iterable.
        geometry_color_mix (float): Set to 1 to use the color provided in the Geometry, 0 to use the color
          specified in the material, or in the range (0,1) to mix the two colors.

    TODO: Document SRGB and linear color spaces, and attributes
    """

    def __init__(self, solid=0, color=(0,0,0), geometry_color_mix=0):
        self._material = _common.Material();

        self.solid = solid;
        self.color = color;
        self.geometry_color_mix = geometry_color_mix;

    @property
    def solid(self):
        return self._material.solid;

    @solid.setter
    def solid(self, value):
        self._material.solid = float(value);

    @property
    def geometry_color_mix(self):
        return self._material.geometry_color_mix;

    @geometry_color_mix.setter
    def geometry_color_mix(self, value):
        self._material.geometry_color_mix = float(value);

    @property
    def color(self):
        return (self._material.color.r, self._material.color.g, self._material.color.b);

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3");
        self._material.color = _common.RGBf(value[0], value[1], value[2]);

    def _get_cpp_material(self):
        return self._material;

class _material_proxy:
    """ Proxy :py:class`Material` attached to a :py:class`fresnel.geometry.Geometry`
    """
    def __init__(self, geometry):
        self._geometry = geometry._geometry;

    @property
    def solid(self):
        m = self._geometry.getMaterial();
        return m.solid;

    @solid.setter
    def solid(self, value):
        m = self._geometry.getMaterial();
        m.solid = float(value);
        self._geometry.setMaterial(m);

    @property
    def geometry_color_mix(self):
        m = self._geometry.getMaterial();
        return m.geometry_color_mix;

    @geometry_color_mix.setter
    def geometry_color_mix(self, value):
        m = self._geometry.getMaterial();
        m.geometry_color_mix = float(value);
        self._geometry.setMaterial(m);

    @property
    def color(self):
        m = self._geometry.getMaterial();
        return (m.color.r, m.color.g, m.color.b);

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3");

        m = self._geometry.getMaterial();
        m.color = _common.RGBf(value[0], value[1], value[2]);
        self._geometry.setMaterial(m);

    def _get_cpp_material(self):
        return self._geometry.getMaterial();

class _outline_material_proxy:
    """ Proxy outline :py:class`Material` attached to a :py:class`fresnel.geometry.Geometry`
    """
    def __init__(self, geometry):
        self._geometry = geometry._geometry;

    @property
    def solid(self):
        m = self._geometry.getOutlineMaterial();
        return m.solid;

    @solid.setter
    def solid(self, value):
        m = self._geometry.getOutlineMaterial();
        m.solid = float(value);
        self._geometry.setOutlineMaterial(m);

    @property
    def geometry_color_mix(self):
        m = self._geometry.getOutlineMaterial();
        return m.geometry_color_mix;

    @geometry_color_mix.setter
    def geometry_color_mix(self, value):
        m = self._geometry.getOutlineMaterial();
        m.geometry_color_mix = float(value);
        self._geometry.setOutlineMaterial(m);

    @property
    def color(self):
        m = self._geometry.getOutlineMaterial();
        return (m.color.r, m.color.g, m.color.b);

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3");

        m = self._geometry.getOutlineMaterial();
        m.color = _common.RGBf(value[0], value[1], value[2]);
        self._geometry.setOutlineMaterial(m);

    def _get_cpp_material(self):
        return self._geometry.getOutlineMaterial();
