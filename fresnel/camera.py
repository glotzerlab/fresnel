# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Cameras.
"""

from . import _common

class Orthographic(object):
    R""" Orthographic camera

    Args:
        position (tuple): :math:`\vec{p}`, the position of the camera.
        look_at (tuple): :math:`\vec{l}`, the point the camera points at.
        up (tuple): :math:`\vec{u}`, the vector pointing up.
        height: `h`, the height of the screen in world distance units.

    :py:class:`Orthographic` defines the parameters of an orthographic camera.

    An orthographic camera is defined by a center position :math:`\vec{p}`, directions :math:`\vec{d}`, :math:`\vec{u}`, and :math:`\vec{r}`,
    and a height `h`. For convenience, this API accepts a look at position :math:`\vec{l}`. Then, :math:`\vec{d}` is the normal
    vector that points from :math:`\vec{p}` to :math:`\vec{l}`.

    The vectors :math:`\vec{d}`, :math:`\vec{u}`, and :math:`\vec{r}` form a right handed orthonormal coordinate system. The vector
    :math:`\vec{u}` points "up" on the screen and :math:`\vec{r}` points to the right. For convenience, you may provide vectors of
    any length in the approximate desired direction. On construction, :py:class:`Orthographic` will normalize vectors
    and form an orthonormal set. The up vector need not be perpendicular to :math:`\vec{d}`, but it must not be parallel.

    In a standard coordinate system, imagine the screen with the x direction pointing right (:math:`\vec{r}`), the
    y direction pointing up (:math:`\vec{u}`), and z pointing *out* of the screen (:math:`-\vec{d}`). With such a camera at 0,0,0,
    only objects at negative z would be visible.

    """
    def __init__(self, position, look_at, up, height):
        u = (look_at[0] - position[0], look_at[1] - position[1], look_at[2] - position[2]);

        self._camera = _common.Camera(_common.vec3f(*position),
                                      _common.vec3f(*u),
                                      _common.vec3f(*up),
                                      height);
