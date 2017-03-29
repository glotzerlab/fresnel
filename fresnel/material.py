# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Materials describe the way light interacts with surfaces.
"""

from fresnel import _common

class Material(object):
    R"""Define material properties.

    Args:

        solid (float): Set to 1 to pass through a solid color, regardless of the light and view angle.
        color (tuple): The linear RGB color of the material as a 3-tuple, list or other iterable.
        primitive_color_mix (float): Set to 1 to use the color provided in the Geometry, 0 to use the color
          specified in the material, or a value in the range [0,1] to mix the two colors.
        roughness (float): Roughness of the material. Nominally in the range [0,1], though 0.1 is a realistic minimum.
        specular (float): Control the strength of the specular highlights. Nominally in the range [0,1].
        metal (float): Set to 0 for dielectric material, or 1 for metal. Intermediate values interpolate between
                       the two.

    Colors are in the linearized sRGB color space. Use :py:func:`fresnel.color.linear` to convert standard sRGB colors
    into this space.
    """

    def __init__(self, solid=0, color=(0,0,0), primitive_color_mix=0, roughness=0.3, specular=0.5, metal=0):
        self._material = _common.Material();

        self.solid = solid;
        self.color = color;
        self.roughness = roughness;
        self.specular = specular;
        self.metal = metal;
        self.primitive_color_mix = primitive_color_mix;

    def __repr__(self):
        return "Material(solid={0}, color={1}, primitive_color_mix={2})".format(self.solid, self.color, self.primitive_color_mix);

    @property
    def solid(self):
        return self._material.solid;

    @solid.setter
    def solid(self, value):
        self._material.solid = float(value);

    @property
    def primitive_color_mix(self):
        return self._material.primitive_color_mix;

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        self._material.primitive_color_mix = float(value);

    @property
    def color(self):
        return (self._material.color.r, self._material.color.g, self._material.color.b);

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3");
        self._material.color = _common.RGBf(*value);

    @property
    def roughness(self):
        return self._material.roughness;

    @roughness.setter
    def roughness(self, value):
        self._material.roughness = float(value);

    @property
    def specular(self):
        return self._material.specular;

    @specular.setter
    def specular(self, value):
        self._material.specular = float(value);

    @property
    def metal(self):
        return self._material.metal;

    @metal.setter
    def metal(self, value):
        self._material.metal = float(value);

    def _get_cpp_material(self):
        return self._material;

class _material_proxy(object):
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
    def primitive_color_mix(self):
        m = self._geometry.getMaterial();
        return m.primitive_color_mix;

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        m = self._geometry.getMaterial();
        m.primitive_color_mix = float(value);
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
        m.color = _common.RGBf(*value);
        self._geometry.setMaterial(m);

    @property
    def roughness(self):
        m = self._geometry.getMaterial();
        return m.roughness;

    @roughness.setter
    def roughness(self, value):
        m = self._geometry.getMaterial();
        m.roughness = float(value);
        self._geometry.setMaterial(m);

    @property
    def specular(self):
        m = self._geometry.getMaterial();
        return m.specular;

    @specular.setter
    def specular(self, value):
        m = self._geometry.getMaterial();
        m.specular = float(value);
        self._geometry.setMaterial(m);

    @property
    def metal(self):
        m = self._geometry.getMaterial();
        return m.metal;

    @metal.setter
    def metal(self, value):
        m = self._geometry.getMaterial();
        m.metal = float(value);
        self._geometry.setMaterial(m);

    def _get_cpp_material(self):
        return self._geometry.getMaterial();

class _outline_material_proxy(object):
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
    def primitive_color_mix(self):
        m = self._geometry.getOutlineMaterial();
        return m.primitive_color_mix;

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        m = self._geometry.getOutlineMaterial();
        m.primitive_color_mix = float(value);
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
        m.color = _common.RGBf(*value);
        self._geometry.setOutlineMaterial(m);

    @property
    def roughness(self):
        m = self._geometry.getOutlineMaterial();
        return m.roughness;

    @roughness.setter
    def roughness(self, value):
        m = self._geometry.getOutlineMaterial();
        m.roughness = float(value);
        self._geometry.setOutlineMaterial(m);

    @property
    def specular(self):
        m = self._geometry.getOutlineMaterial();
        return m.specular;

    @specular.setter
    def specular(self, value):
        m = self._geometry.getOutlineMaterial();
        m.specular = float(value);
        self._geometry.setOutlineMaterial(m);

    @property
    def metal(self):
        m = self._geometry.getOutlineMaterial();
        return m.metal;

    @metal.setter
    def metal(self, value):
        m = self._geometry.getOutlineMaterial();
        m.metal = float(value);
        self._geometry.setOutlineMaterial(m);

    def _get_cpp_material(self):
        return self._geometry.getOutlineMaterial();
