# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Materials describe the way light interacts with surfaces.
"""

class Material:
    R"""Define material properties.

    Args:

        solid (float): Set to 1 to pass through a solid color, regardless of the light and view angle.
        color (tuple): 3-tuple, list or other iterable that specifies the RGB color of the material.

    TODO: Document SRGB and linear color spaces.
    """

    def __init__(self, solid=0, color=(0,0,0)):
        self.solid = solid;
        self.color = color;
