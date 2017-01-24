# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Color utilities.
"""

import numpy

def linear(color):
    R""" Convert a sRGB color from the gamma corrected color space into the linear space.

    RGB colors, such as color pickers in most applications, are usually provided in the sRGB color space with gamma
    correction. fresnel needs to perform calculations in a linear color space. This method converts a color into
    that linear space.

    Args:

        color (tuple): 3-length list, or other object convertible to a numpy array.

    Returns:

        A length 3 numpy array with the linearized color.
    """

    c = numpy.ascontiguousarray(color);
    if len(c) != 3:
        raise TypeError("color must be a length 3 array");

    out = numpy.zeros(3, dtype=numpy.float32)
    if c[0] < 0.04045:
        out[0] = c[0] / 12.92;
    else:
        out[0] = ((c[0] + 0.055) / (1.055))**2.4;

    if c[1] < 0.04045:
        out[1] = c[1] / 12.92;
    else:
        out[1] = ((c[1] + 0.055) / (1.055))**2.4;

    if c[2] < 0.04045:
        out[2] = c[2] / 12.92;
    else:
        out[2] = ((c[2] + 0.055) / (1.055))**2.4;

    return out;
