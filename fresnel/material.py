# Copyright (c) 2016-2021 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""Materials describe the way light interacts with surfaces."""

from . import _common


class Material(object):
    """Define material properties.

    Materials control how light interacts with the geometry.

    Args:
        solid (float): Set to 1 to pass through a solid color, regardless of the
            light and view angle.

        color ((3, ) `numpy.ndarray` of ``float32``)): Linear material color.

        primitive_color_mix (float): Set to 1 to use the color provided in the
            `Geometry`, 0 to use the color specified in the `Material`, or a
            value in the range [0, 1] to mix the two colors.

        roughness (float): Roughness of the material. Nominally in the range
            [0.1, 1].

        specular (float): Control the strength of the specular highlights.
            Nominally in the range [0, 1].

        spec_trans (float): Control the amount of specular light transmission.
            In the range [0, 1].

        metal (float): Set to 0 for dielectric material, or 1 for metal.
            Intermediate values interpolate between the two.

    See Also:
        Tutorials:

        - :doc:`examples/00-Basic-tutorials/02-Material-properties`

    Note:
        Colors are in the linearized color space. Use `fresnel.color.linear` to
        convert standard sRGB colors into this space.
    """

    def __init__(self,
                 solid=0,
                 color=(0.9, 0.9, 0.9),
                 primitive_color_mix=0,
                 roughness=0.3,
                 specular=0.5,
                 spec_trans=0,
                 metal=0):
        self._material = _common.Material()

        self.solid = solid
        self.color = color
        self.roughness = roughness
        self.specular = specular
        self.metal = metal
        self.spec_trans = spec_trans
        self.primitive_color_mix = primitive_color_mix

    @property
    def solid(self):
        """float: Is this material a solid color?

        Set to 1 to pass through a solid color, regardless of the light and view
        angle.
        """
        return self._material.solid

    @solid.setter
    def solid(self, value):
        self._material.solid = float(value)

    @property
    def primitive_color_mix(self):
        """float: Mix the material color with the geometry.

        Set to 1 to use the color provided in the `Geometry`, 0 to use the color
        specified in the `Material`, or a value in the range [0, 1] to mix
        the two colors.
        """
        return self._material.primitive_color_mix

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        self._material.primitive_color_mix = float(value)

    @property
    def color(self):
        """((3, ) `numpy.ndarray` of ``float32``)): - Linear material color."""
        return (self._material.color.r, self._material.color.g,
                self._material.color.b)

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3")
        self._material.color = _common.RGBf(*value)

    @property
    def roughness(self):
        """float: Roughness of the material.

        Nominally in the range [0.1, 1].
        """
        return self._material.roughness

    @roughness.setter
    def roughness(self, value):
        self._material.roughness = float(value)

    @property
    def specular(self):
        """float: Control the strength of the specular highlights.

        Nominally in the range [0, 1].
        """
        return self._material.specular

    @specular.setter
    def specular(self, value):
        self._material.specular = float(value)

    @property
    def spec_trans(self):
        """float: Control the amount of specular light transmission.

        In the range [0, 1].
        """
        return self._material.spec_trans

    @spec_trans.setter
    def spec_trans(self, value):
        self._material.spec_trans = float(value)

    @property
    def metal(self):
        """float: Set to 0 for dielectric material, or 1 for metal.

        Intermediate values interpolate between the two.
        """
        return self._material.metal

    @metal.setter
    def metal(self, value):
        self._material.metal = float(value)

    def _get_cpp_material(self):
        return self._material


class _MaterialProxy(object):
    """Proxy `Material` attached to a `Geometry`."""

    def __init__(self, geometry):
        self._geometry = geometry._geometry

    @property
    def solid(self):
        m = self._geometry.getMaterial()
        return m.solid

    @solid.setter
    def solid(self, value):
        m = self._geometry.getMaterial()
        m.solid = float(value)
        self._geometry.setMaterial(m)

    @property
    def primitive_color_mix(self):
        m = self._geometry.getMaterial()
        return m.primitive_color_mix

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        m = self._geometry.getMaterial()
        m.primitive_color_mix = float(value)
        self._geometry.setMaterial(m)

    @property
    def color(self):
        m = self._geometry.getMaterial()
        return (m.color.r, m.color.g, m.color.b)

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3")

        m = self._geometry.getMaterial()
        m.color = _common.RGBf(*value)
        self._geometry.setMaterial(m)

    @property
    def roughness(self):
        m = self._geometry.getMaterial()
        return m.roughness

    @roughness.setter
    def roughness(self, value):
        m = self._geometry.getMaterial()
        m.roughness = float(value)
        self._geometry.setMaterial(m)

    @property
    def specular(self):
        m = self._geometry.getMaterial()
        return m.specular

    @specular.setter
    def specular(self, value):
        m = self._geometry.getMaterial()
        m.specular = float(value)
        self._geometry.setMaterial(m)

    @property
    def spec_trans(self):
        m = self._geometry.getMaterial()
        return m.spec_trans

    @spec_trans.setter
    def spec_trans(self, value):
        m = self._geometry.getMaterial()
        m.spec_trans = float(value)
        self._geometry.setMaterial(m)

    @property
    def metal(self):
        m = self._geometry.getMaterial()
        return m.metal

    @metal.setter
    def metal(self, value):
        m = self._geometry.getMaterial()
        m.metal = float(value)
        self._geometry.setMaterial(m)

    def _get_cpp_material(self):
        return self._geometry.getMaterial()


class _OutlineMaterialProxy(object):
    """Proxy outline `Material` attached to a Geometry."""

    def __init__(self, geometry):
        self._geometry = geometry._geometry

    @property
    def solid(self):
        m = self._geometry.getOutlineMaterial()
        return m.solid

    @solid.setter
    def solid(self, value):
        m = self._geometry.getOutlineMaterial()
        m.solid = float(value)
        self._geometry.setOutlineMaterial(m)

    @property
    def primitive_color_mix(self):
        m = self._geometry.getOutlineMaterial()
        return m.primitive_color_mix

    @primitive_color_mix.setter
    def primitive_color_mix(self, value):
        m = self._geometry.getOutlineMaterial()
        m.primitive_color_mix = float(value)
        self._geometry.setOutlineMaterial(m)

    @property
    def color(self):
        m = self._geometry.getOutlineMaterial()
        return (m.color.r, m.color.g, m.color.b)

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("colors must have length 3")

        m = self._geometry.getOutlineMaterial()
        m.color = _common.RGBf(*value)
        self._geometry.setOutlineMaterial(m)

    @property
    def roughness(self):
        m = self._geometry.getOutlineMaterial()
        return m.roughness

    @roughness.setter
    def roughness(self, value):
        m = self._geometry.getOutlineMaterial()
        m.roughness = float(value)
        self._geometry.setOutlineMaterial(m)

    @property
    def specular(self):
        m = self._geometry.getOutlineMaterial()
        return m.specular

    @specular.setter
    def specular(self, value):
        m = self._geometry.getOutlineMaterial()
        m.specular = float(value)
        self._geometry.setOutlineMaterial(m)

    @property
    def spec_trans(self):
        m = self._geometry.getOutlineMaterial()
        return m.spec_trans

    @spec_trans.setter
    def spec_trans(self, value):
        m = self._geometry.getOutlineMaterial()
        m.spec_trans = float(value)
        self._geometry.setOutlineMaterial(m)

    @property
    def metal(self):
        m = self._geometry.getOutlineMaterial()
        return m.metal

    @metal.setter
    def metal(self, value):
        m = self._geometry.getOutlineMaterial()
        m.metal = float(value)
        self._geometry.setOutlineMaterial(m)

    def _get_cpp_material(self):
        return self._geometry.getOutlineMaterial()
