# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""Utilities."""

import numpy
import io

try:
    import PIL.Image as PIL_Image
except ImportError:
    PIL_Image = None


class Array(object):
    """Access fresnel memory buffers.

    `Array` provides a python interface to access the internal data of memory
    buffers stored and managed by fresnel. You can access a `Array` as if it
    were a `numpy.ndarray` (with limited operations). Below, *slice* is a
    :std:term:`slice` or array `indexing <numpy.doc.indexing>` mechanic that
    **numpy** understands.

    .. rubric:: Writing

    Write to an array with ``array[slice] = v`` where *v* is
    `numpy.ndarray`, `list`, or scalar value to broadcast. When
    *v* is a *contiguous* `numpy.ndarray` of the same data type, the
    data is copied directly from *v* into the internal buffer. Otherwise, it is
    converted to a `numpy.ndarray` before copying.

    .. rubric:: Reading

    Read from an array with ``v = array[slice]``. This returns a **copy** of the
    data as a `numpy.ndarray` each time it is called.

    Attributes:
        shape (tuple[int, [int]]): Dimensions of the array.
        dtype: Numpy data type
    """

    def __init__(self, buf, geom):
        self.buf = buf
        # geom stores a pointer to the owning geometry, so array writes trigger
        # acceleration structure updates set to None if this buffer is not
        # associated with a geometry
        self.geom = geom

        self.buf.map()
        a = numpy.array(self.buf, copy=False)
        self.shape = a.shape
        self.dtype = a.dtype
        self.buf.unmap()

    def __setitem__(self, slice, data):
        """Assign a data array to a slice."""
        self.buf.map()
        a = numpy.array(self.buf, copy=False)
        a[slice] = data
        self.buf.unmap()

        if self.geom is not None:
            self.geom._geometry.update()

    def __getitem__(self, slice):
        """Make a copy of the data in the buffer."""
        self.buf.map()
        a = numpy.array(self.buf, copy=False)
        data = numpy.array(a[slice], copy=True)
        self.buf.unmap()
        return data


class ImageArray(Array):
    """Access fresnel images.

    Provide `Array` functionality withsome additional convenience methods
    specific to working with images. Images are represented as ``(W, H, 4)``
    `numpy.ndarray` of ``uint8`` values in **RGBA** format.

    When a `ImageArray` is the result of an image in a Jupyter notebook cell,
    Jupyter will display the image.
    """

    def _repr_png_(self):
        if PIL_Image is None:
            raise RuntimeError("No PIL.Image module to format png")

        self.buf.map()

        f = io.BytesIO()
        a = numpy.array(self.buf, copy=False)
        PIL_Image.fromarray(a, mode='RGBA').save(f, 'png')
        self.buf.unmap()

        return f.getvalue()


def convex_polyhedron_from_vertices(vertices):
    """Make a convex polyhedron from vertices.

    Args:
        vertices ((3, ) `numpy.ndarray` of ``float32``): Vertices of the\
            polyhedron.

    Returns:
        dict: Convex hull of *vertices* in a format used by `ConvexPolyhedron`.

        The dictionary contains the keys ``face_origin``, ``face_normal``,
        ``face_color``, and ``radius``.

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
