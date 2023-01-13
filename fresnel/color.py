# Copyright (c) 2016-2023 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Color utilities."""

import numpy


def linear(color):
    """Convert a sRGB color (or colors) into the linear space.

    Standard tools for working with `sRGB <https://en.wikipedia.org/wiki/SRGB>`_
    colors provide gamma corrected values. fresnel needs to perform calculations
    in a linear color space. This method converts from sRGB to the linear space.
    Use :py:func:`linear` when specifying material or particle colors with sRGB
    inputs (such as you find in a color picker).

    :py:func:`linear` accepts
    `RGBA <https://en.wikipedia.org/wiki/RGBA_color_space>`_ input (such as from
    matplotlib's `colors.to_rgba
    <https://matplotlib.org/api/_as_gen/matplotlib.colors.to_rgba.html>`_
    colormap method), but ignores the alpha channel and outputs an ``Nx3``
    array.

    Args:
        color ((3, ), (4, ), (N, 3), or (N, 4) `numpy.ndarray` of\
            ``numpy.float32``): ``RGB`` or ``RGBA`` colors.

    Color components are in the range [0,1].

    Returns:
        `numpy.ndarray` with the linearized color(s), same shape as ``color``.
    """
    c = numpy.ascontiguousarray(color)
    if c.shape == (3,) or c.shape == (4,):
        out = numpy.zeros(3, dtype=numpy.float32)
        if c[0] < 0.04045:
            out[0] = c[0] / 12.92
        else:
            out[0] = ((c[0] + 0.055) / (1.055))**2.4

        if c[1] < 0.04045:
            out[1] = c[1] / 12.92
        else:
            out[1] = ((c[1] + 0.055) / (1.055))**2.4

        if c[2] < 0.04045:
            out[2] = c[2] / 12.92
        else:
            out[2] = ((c[2] + 0.055) / (1.055))**2.4

    elif c.ndim == 2 and (c.shape[1] == 3 or c.shape[1] == 4):
        out = numpy.zeros(shape=(c.shape[0], 3), dtype=numpy.float32)
        for i in range(3):
            s = c[:, i] < 0.04045
            out[s, i] = c[s, i] / 12.92
            not_s = numpy.logical_not(s)
            out[not_s, i] = ((c[not_s, i] + 0.055) / (1.055))**2.4
    else:
        raise TypeError("color must be a length 3, 4, Nx3, or Nx4 array")

    return out
