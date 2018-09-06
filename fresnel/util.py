# Copyright (c) 2016-2018 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Utility classes and methods.
"""

import numpy
import io
import itertools

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

    @property
    def shape(self):
        self.buf.map();
        a = numpy.array(self.buf, copy=False);
        self.buf.unmap();
        return a.shape;

    @property
    def dtype(self):
        self.buf.map();
        a = numpy.array(self.buf, copy=False);
        self.buf.unmap();
        return a.dtype;

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


def convex_polyhedron_from_vertices(vertices):
    R""" Get origins and normals for a convex polyhedron for its vertices

    Args:
        vertices (array-like, shape=(n,3)): The vertices of the polyhedron

    Returns:
        The ``origins``, ``normals``, and ``r`` to pass to :py:class:`fresnel.geometry.ConvexPolyhedron`.

    This function is intended to be used to draw a convex polygon given its vertices. It
    returns ``origins``, ``normals``, and ``r`` that can be passed when drawing a convex
    polyhedron:

    .. highlight:: python
    .. code-block:: python

        origins, normals, r_circ = convex_polyhedron_from_vertices(vertices)
        fresnel.geometry.ConvexPolyhedron(scene, origins, normals, r_circ)
    """
    from scipy.spatial import ConvexHull

    ch = ConvexHull(vertices)
    origins = -ch.equations[:, :-1] * numpy.tile(ch.equations[:, -1], (3, 1)).T
    normals = ch.equations[:, :-1]
    merged_origins, merged_normals = [], []
    origin_combos = itertools.combinations(origins, 2)
    normal_combos = itertools.combinations(normals, 2)
    for ((a, b), (c, d)) in zip(origin_combos, normal_combos):
        if numpy.isclose(numpy.dot(c, d), 1):
            merged_origins.append(a)
            merged_normals.append(c)
    r = _get_r_circ(vertices)
    return numpy.array(merged_origins), numpy.array(merged_normals), r


def _get_r_circ(vertices):
    """Estimate circumsphere radius based on vertices of a polyhedron
    """
    vertices = numpy.array(vertices)
    radii = numpy.sqrt(numpy.sum(vertices**2, axis=1))
    return numpy.amax(radii)
