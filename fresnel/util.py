# Copyright (c) 2016-2019 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Utilities.
"""

import numpy
import io


try:
    import PIL.Image as PIL_Image;
except ImportError:
    PIL_Image = None;

class array(object):
    R""" Map internal fresnel buffers as :py:class:`numpy.ndarray` objects.

    :py:class:`fresnel.util.array` provides a python interface to access internal data of memory buffers stored and
    managed by fresnel. You can access a :py:class:`fresnel.util.array` as if it were a :py:class:`numpy.ndarray` (with
    limited operations). Below, *slice* is a :std:term:`slice` or array `indexing <numpy.doc.indexing>` mechanic that
    **numpy** understands.

    .. rubric:: Writing

    Write to an array with ``array[slice] = v`` where *v* is :py:class:`numpy.ndarray`, :any:`list`, or
    scalar value to broadcast. When *v* is a *contiguous* :py:class:`numpy.ndarray` of the same data type, the data is
    copied directly from *v* into the internal buffer. Otherwise, it is converted to a :py:class:`numpy.ndarray`
    before copying.

    .. rubric:: Reading

    Read from an array with ``v = array[slice]``. This returns a **copy** of the data as a :py:class:`numpy.ndarray`
    each time it is called.

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
    R""" Map internal image buffers as :py:class:`numpy.ndarray` objects.

    Inherits from :py:class:`array` and provides all of its functionality, plus some additional convenience methods
    specific to working with images. Images are represented as ``WxHx4`` :py:class:`numpy.ndarray` of ``uint8`` values
    in **RGBA** format.

    When a :py:class:`image_array` is the result of an image in a Jupyter notebook cell, Jupyter will
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


def convex_polyhedron_from_vertices(vertices):
    R""" Convert convex polyhedron vertices to data structures that fresnel can draw.

    Args:
        vertices (`numpy.ndarray` or `array_like`): (``Nx3`` : ``float64``) - The vertices of the polyhedron.

    Returns:
        A dict containing the information used to draw the polyhedron. The dict
        contains the keys ``face_origin``, ``face_normal``, ``face_color``, and ``radius``.

    The dictionary can be used directly to draw a polyhedron from its vertices:

    .. highlight:: python
    .. code-block:: python

        scene = fresnel.Scene()
        polyhedron = fresnel.util.convex_polyhedron_from_vertices(vertices)
        geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                     polyhedron,
                                                     position=[0, 0, 0],
                                                     orientation=[1, 0, 0, 0])

    """
    from fresnel._common import find_polyhedron_faces
    # sanity checks on the shape of things here?
    return find_polyhedron_faces(vertices)
