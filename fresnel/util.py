# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Utility classes and methods.
"""

import numpy
import io

try:
    import PIL.Image as PIL_Image;
except ImportError:
    PIL_Image = None;

class array(object):
    R""" Map internal fresnel buffers as numpy arrays.

    :py:class:`fresnel.util.array` provides a python interface to access internal data of memory buffers stored and
    managed by fresnel. These buffers may exist on the CPU or GPU depending on the device configuration, so
    :py:class:`fresnel.util.array` only allows certain operations: read/write of array data, and read-only querying of
    array properties.

    You can access a :py:class:`fresnel.util.array` as if it were a numpy array (with limited operations).

    Write to an array with ``array[slice] = v`` where *v* is a numpy array or anything that numpy can convert to an
    array. When *v* is a contiguous numpy array of the appropriate data type, the data is copied directly from *v*
    into the internal buffer.

    Read from an array with ``v = array[slice]``. This returns a **copy** of the data as a numpy array because the
    array references internal data structures in fresnel that may exist on the GPU.

    Attributes:

        shape (tuple): Dimensions of the array.
        dtype: Numpy data type
    """

    def __init__(self, buf, geom):
        self.buf = buf;
        # geom stores a pointer to the owning geometry, so array writes trigger acceleration structure updates
        # set to None if this buffer is not associated with a geometry
        self.geom = geom

        self.buf.map();
        a = numpy.array(self.buf, copy=False);
        self.shape = a.shape;
        self.dtype = a.dtype;
        self.buf.unmap();

    def __setitem__(self, slice, data):
        self.buf.map();
        a = numpy.array(self.buf, copy=False);
        a[slice] = data;
        self.buf.unmap();

        if self.geom is not None:
            self.geom._geometry.update();

    def __getitem__(self, slice):
        self.buf.map();
        a = numpy.array(self.buf, copy=False);
        data = numpy.array(a[slice], copy=True);
        self.buf.unmap();
        return data;

class image_array(array):
    R""" Map internal fresnel image buffers as numpy arrays.

    Inherits from :py:class:`array` and provides all of its functionality, plus some additional convenience methods
    specific to working with images. Images are represented as WxHx4 numpy arrays of unsigned chars in RGBA format.

    Specifically, when a :py:class:`image_array` is the result of an image in a Jupyter notebook cell, Jupyter will
    display the image.
    """

    def _repr_png_(self):
        if PIL_Image is None:
            raise RuntimeError("No PIL.Image module to format png");

        self.buf.map();

        f = io.BytesIO();
        a = numpy.array(self.buf, copy=False);
        PIL_Image.fromarray(a, mode='RGBA').save(f, 'png');
        self.buf.unmap();

        return f.getvalue();
