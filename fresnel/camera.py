# Copyright (c) 2016-2018 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Cameras.
"""

import numpy
import math

from . import _common

class Camera(object):
    R""" Camera

    Defines the view into the :py:class:`Scene <fresnel.Scene>`.

    Use one of the creation functions to create a :py:class:`Camera <fresnel.camera.Camera>`:

        * :py:func:`orthographic`

    The camera is a property of the :py:class:`Scene <fresnel.Scene>`. You may read and modify any of these camera attributes.

    Attributes:
        position (tuple): the position of the camera (the center of projection).
        look_at (tuple): the point the camera looks at (the center of the focal plane).
        up (tuple): a vector pointing up.
        height: the height of the image plane.

    :py:class:`Camera <fresnel.camera.Camera>` space is a coordinate system centered on the camera's position.
    Positive *x* points to the right in the image, positive *y* points up, and positive *z* points out of the screen.
    :py:class:`Camera <fresnel.camera.Camera>` space shares units with :py:class:`Scene <fresnel.Scene>` space.

    TODO: Move description of spaces to an overview page and create figures.
    """
    def __init__(self, _camera=None):
        if _camera is None:
            self._camera = _common.UserCamera();
        else:
            self._camera = _camera;

    @property
    def position(self):
        return (self._camera.position.x, self._camera.position.y, self._camera.position.z);

    @position.setter
    def position(self, value):
        if len(value) != 3:
            raise ValueError("position must have length 3");
        self._camera.position = _common.vec3f(*value);

    @property
    def look_at(self):
        return (self._camera.look_at.x, self._camera.look_at.y, self._camera.look_at.z);

    @look_at.setter
    def look_at(self, value):
        if len(value) != 3:
            raise ValueError("look_at must have length 3");
        self._camera.look_at = _common.vec3f(*value);

    @property
    def up(self):
        return (self._camera.up.x, self._camera.up.y, self._camera.up.z);

    @up.setter
    def up(self, value):
        if len(value) != 3:
            raise ValueError("up must have length 3");
        self._camera.up = _common.vec3f(*value);

    @property
    def height(self):
        return self._camera.h;

    @height.setter
    def height(self, value):
        self._camera.h = float(value);

    def __repr__(self):
        return "fresnel.camera.orthographic(position={0}, look_at={1}, up={2}, height={3})".format(self.position, self.look_at, self.up, self.height)

    def __str__(self):
        return "<Camera object with position {0}>".format(self.position);

def orthographic(position, look_at, up, height):
    R""" Orthographic camera

    Args:
        position (tuple): the position of the camera.
        look_at (tuple): the point the camera looks at (the center of the focal plane).
        up (tuple): a vector pointing up.
        height: the height of the image plane.

    An orthographic camera traces parallel rays from the image plane into the scene. Lines that are parallel in
    the :py:class:`Scene <fresnel.Scene>` will remain parallel in the rendered image.

    *position* is the center of the image plane in :py:class:`Scene <fresnel.Scene>` space. *look_at* is the point
    in :py:class:`Scene <fresnel.Scene>` space that will be in the center of the image.  Together, these vectors define
    the image plane which is perpendicular to the line from *position* to *look_at*. Objects in front of the plane will
    appear in the rendered image, objects behind the plane will not.

    *up* is a vector in :py:class:`Scene <fresnel.Scene>` space that defines which direction points up (+y direction in the image).
    *up* does not need to be perpendicular to the line from *position* to *look_at*, but it must not be parallel to that
    line. *height* sets the height of the image in :py:class:`Scene <fresnel.Scene>` units. The image width is determined by the
    aspect ratio of the image. The area *width* by *height* about the *look_at* point will be included in the rendered
    image.

    TODO: show a figure
    """

    cam = Camera();
    cam.position = position;
    cam.look_at = look_at;
    cam.up = up;
    cam.height = height;

    return cam

def fit(scene, view='auto', margin=0.05):
    R""" Fit a camera to a :py:class:`Scene <fresnel.Scene>`

    Create a camera that fits the entire hight of the scene in the image plane.

    Args:
        scene (:py:class:`Scene <fresnel.Scene>`): The scene to fit the camera to.
        view (str): Select view
        margin (float): Fraction of extra space to leave on the top and bottom of the scene.

    *view* may be 'auto', 'isometric', or 'front'.

    The isometric view is an orthographic projection from a particular angle so that the x,y, and z directions
    are equal lengths. The front view is an orthographic projection where +x points to the right, +y points up
    and +z points out of the screen in the image plane. 'auto' automatically selects 'isometric' for 3D scenes
    and 'front' for 2D scenes.
    """

    vectors = {'front': dict(v=numpy.array([0,0,1]), up = numpy.array([0,1,0]), right = numpy.array([1,0,0])),
               'isometric': dict(v = numpy.array([1, 1, 1])/math.sqrt(3),
                                 up = numpy.array([-1, 2, -1])/math.sqrt(6),
                                 right = numpy.array([1, 0, -1])/math.sqrt(2))
              }

    # find the center of the scene
    extents = scene.get_extents();

    # choose an appropriate view automatically
    if view == 'auto':
        xw = extents[1,0] - extents[0,0];
        yw = extents[1,1] - extents[0,1];
        zw = extents[1,2] - extents[0,2];

        if zw < 0.51 * max(xw, yw):
            view = 'front';
        else:
            view = 'isometric';

    v = vectors[view]['v'];
    up = vectors[view]['up'];
    right = vectors[view]['right'];

    # make a list of points of the cube surrounding the scene
    points = numpy.array([[extents[0,0], extents[0,1], extents[0,2]],
                          [extents[0,0], extents[0,1], extents[1,2]],
                          [extents[0,0], extents[1,1], extents[0,2]],
                          [extents[0,0], extents[1,1], extents[1,2]],
                          [extents[1,0], extents[0,1], extents[0,2]],
                          [extents[1,0], extents[0,1], extents[1,2]],
                          [extents[1,0], extents[1,1], extents[0,2]],
                          [extents[1,0], extents[1,1], extents[1,2]]]);

    # find the center of the box
    center = (extents[0,:] + extents[1,:])  / 2;
    points = points - center;

    # determine the extent of the scene box in the up direction
    up_projection = numpy.dot(points, up);
    height = (1+margin)*numpy.max(numpy.abs(up_projection))*2;

    # determine the extent of the scene box in the view direction
    view_projection = numpy.dot(points, v);
    view_distance = numpy.max(view_projection) * 1.10;

    # build the camera
    return orthographic(position = center+view_distance*v,
                        look_at = center,
                        up = up,
                        height = height);

