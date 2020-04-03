# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

R"""
Cameras.
"""

import numpy
import math

from . import _common


class Camera(object):
    R""" Camera

    Defines the view into the :py:class:`Scene <fresnel.Scene>`.

    Use one of the creation functions to create a
    :py:class:`Camera <fresnel.camera.Camera>`:

        * :py:func:`orthographic`

    .. seealso::
        Tutorials:

        - :doc:`examples/00-Basic-tutorials/04-Scene-properties`

    The camera is a property of the :py:class:`Scene <fresnel.Scene>`. You may
    read and modify any of these camera attributes.

    Attributes:
        position (tuple[float, float, float]): the position of the camera (the
            center of projection).
        look_at (tuple[float, float, float]): the point the camera looks at (the
            center of the focal plane).
        up (tuple[float, float, float]): a vector pointing up.
        height (float): the height of the image plane.
        basis: three orthonormal vectors defining the camera coordinate basis in
            right-handed order: right, look direction, up (read only)

    :py:class:`Camera <fresnel.camera.Camera>` space is a coordinate system
    centered on the camera's position. Positive *x* points to the right in the
    image, positive *y* points up, and positive *z* points out of the screen.
    :py:class:`Camera <fresnel.camera.Camera>` space shares units with
    :py:class:`Scene <fresnel.Scene>` space.

    """

    def __init__(self, _camera=None):
        if _camera is None:
            self._camera = _common.UserCamera()
            self.position = (0, 0, 0)
            self.look_at = (0, 0, 1)
            self.up = (0, 1, 0)
            self.height = 1
            self.mode = 'orthographic'
        else:
            self._camera = _camera

    @property
    def position(self):
        return (self._camera.position.x,
                self._camera.position.y,
                self._camera.position.z)

    @position.setter
    def position(self, value):
        if len(value) != 3:
            raise ValueError("position must have length 3")
        self._camera.position = _common.vec3f(*value)

    @property
    def look_at(self):
        return (self._camera.look_at.x,
                self._camera.look_at.y,
                self._camera.look_at.z)

    @look_at.setter
    def look_at(self, value):
        if len(value) != 3:
            raise ValueError("look_at must have length 3")
        self._camera.look_at = _common.vec3f(*value)

    @property
    def up(self):
        return (self._camera.up.x, self._camera.up.y, self._camera.up.z)

    @up.setter
    def up(self, value):
        if len(value) != 3:
            raise ValueError("up must have length 3")
        self._camera.up = _common.vec3f(*value)

    @property
    def height(self):
        return self._camera.h

    @height.setter
    def height(self, value):
        self._camera.h = float(value)

    @property
    def focal_length(self):
        """Focal length of the camera lens.

        note:
            `focal_length` is only used with the `pinhole` and `thin_lens`
            models.
        """
        return self._camera.f

    @focal_length.setter
    def focal_length(self, value):
        self._camera.f = value

    @property
    def f_stop(self):
        """F-stop value for the lens.

        Set the aperture of the opening into the lens in f-stops. This controls
        the depth of field in the ``thin_lens`` model.

        Note:
            :py:attr:`f_stop` is only used with the ``thin_lens``
            :py:attr:`model`.

        See:
            :py:attr:`depth_of_field`
        """
        return self._camera.f_stop

    @f_stop.setter
    def f_stop(self, value):
        self._camera.f_stop = value

    @property
    def basis(self):
        b = _common.CameraBasis(self._camera)
        return ((b.u.x, b.u.y, b.u.z),
                (b.v.x, b.v.y, b.v.z),
                (b.w.x, b.w.y, b.w.z))

    @property
    def model(self):
        """The camera type to model.

        Valid values are:

        * "orthographic"
        * "pinhole"
        * "thin_lens"
        """
        return self._camera.model.name

    @model.setter
    def model(self, value):
        self._camera.model = getattr(_common.CameraModel, value)

    @property
    def subject_distance(self):
        """Distance to the subject.

        The subject distance is the distance from the camera position
        (:py:attr:`position`) to the center of thefocal plane
        (:py:attr:`look_at`).

        Setting :py:attr:`subject_distance` will modify :py:attr:`look_at`
        to match the given distance.
        """
        look = numpy.array(self.look_at) - numpy.array(self.position)
        return numpy.sqrt(numpy.dot(look, look))

    @subject_distance.setter
    def subject_distance(self, value):
        look_at = numpy.array(self.look_at)
        position = numpy.array(self.position)

        # form a normal vector in the direction of the camera
        look = look_at - position
        look /= numpy.sqrt(numpy.dot(look, look))

        # move look_at to the new distance
        self.look_at = position + value * look

    @property
    def depth_of_field(self):
        """The distance about the focal plane in sharp focus.

        Note:
            Depth of field only applies when using the ``thin_lens`` model.

        The area of sharp focus is centered at the :py:attr:`look_at` point
        and extends in front of and behind the focal plane. The distance
        between the front and back areas of sharp focus is the depth of field.

        The depth of field is a function of by the subject distance (the
        distance between :py:attr:`position` and :py:attr:`look_at`), the focal
        length of the lens (:py:attr:`focal_length`), the size of the aperture
        (:py:attr:`f_stop`), and the height of the image plane
        (:py:attr:`height`).

        Note:
            Changing any one of these parameters can lead to a dramatic change
            in the depth of field.

        Setting :py:attr:`depth_of_field` sets :py:attr:`f_stop` to obtain
        the desired depth of field as a function of the current values of the
        other parameter values. If you later change the other camera properties,
        the depth of field will change as well.
        """

        f = self.focal_length
        N = self.f_stop
        c = self.height / 720
        s = self.subject_distance

        H = f**2 / (N * c)
        return s * H * (1 / (H - (s - f)) - 1 / (H + (s - f)))

    @depth_of_field.setter
    def depth_of_field(self, value):
        f = self.focal_length
        c = self.height / 720
        d = value
        s = self.subject_distance

        N = (math.sqrt(c**2 * f**4 * (d**2 + s**2) * (f - s)**2)
            + c * f**2 * s * (f - s))/(c**2 * d * (f - s)**2)
        self.f_stop = N


    def __repr__(self):
        s = "fresnel.camera.orthographic("
        s += f"position={self.position}, "
        s += f"look_at={self.look_at}, "
        s += f"up={self.up}, "
        s += f"height={self.height})"
        return s

    def __str__(self):
        return "<Camera object with position {0}>".format(self.position)


def orthographic(position, look_at, up, height):
    R""" Orthographic camera

    Args:
        position (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - the
            position of the camera.
        look_at (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - the
            point the camera looks at (the center of the focal plane).
        up (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - a vector
            pointing up.
        height (float): the height of the image plane.

    An orthographic camera traces parallel rays from the image plane into the
    scene. Lines that are parallel in the :py:class:`Scene <fresnel.Scene>` will
    remain parallel in the rendered image.

    *position* is the center of the image plane in
    :py:class:`Scene <fresnel.Scene>` space. *look_at* is the point in
    :py:class:`Scene <fresnel.Scene>` space that will be in the center of the
    image.  Together, these vectors define the image plane which is
    perpendicular to the line from *position* to *look_at*. Objects in front of
    the plane will appear in the rendered image, objects behind the plane will
    not.

    *up* is a vector in :py:class:`Scene <fresnel.Scene>` space that defines
    which direction points up (+y direction in the image). *up* does not need to
    be perpendicular to the line from *position* to *look_at*, but it must not
    be parallel to that line. *height* sets the height of the image in
    :py:class:`Scene <fresnel.Scene>` units. The image width is determined by
    the aspect ratio of the image. The area *width* by *height* about the
    *look_at* point will be included in the rendered image.

    """

    cam = Camera()
    cam.position = position
    cam.look_at = look_at
    cam.up = up
    cam.height = height
    cam.model = "orthographic"

    return cam


def fit(scene, view='auto', margin=0.05):
    R""" Fit a camera to a :py:class:`Scene <fresnel.Scene>`

    Create a camera that fits the entire height of the scene in the image plane.

    Args:
        scene (:py:class:`Scene <fresnel.Scene>`): Fit the camera to this scene.
        view (str): Select view
        margin (float): Fraction of extra space to leave on the top and bottom
            of the scene.

    *view* may be 'auto', 'isometric', or 'front'.

    The isometric view is an orthographic projection from a particular angle so
    that the x,y, and z directions are equal lengths. The front view is an
    orthographic projection where +x points to the right, +y points up and +z
    points out of the screen in the image plane. 'auto' automatically selects
    'isometric' for 3D scenes and 'front' for 2D scenes.
    """

    vectors = {'front': dict(v=numpy.array([0, 0, 1]),
                             up=numpy.array([0, 1, 0]),
                             right=numpy.array([1, 0, 0])),
               'isometric': dict(v=numpy.array([1, 1, 1]) / math.sqrt(3),
                                 up=numpy.array([-1, 2, -1]) / math.sqrt(6),
                                 right=numpy.array([1, 0, -1]) / math.sqrt(2))
               }

    # raise error if the scene is empty
    if len(scene.geometry) == 0:
        raise ValueError('The camera cannot be fit because the scene has no '
                         'geometries. Add geometries to the scene before '
                         'calling fit.')

    # find the center of the scene
    extents = scene.get_extents()

    # choose an appropriate view automatically
    if view == 'auto':
        xw = extents[1, 0] - extents[0, 0]
        yw = extents[1, 1] - extents[0, 1]
        zw = extents[1, 2] - extents[0, 2]

        if zw < 0.51 * max(xw, yw):
            view = 'front'
        else:
            view = 'isometric'

    v = vectors[view]['v']
    up = vectors[view]['up']
    # right = vectors[view]['right']

    # make a list of points of the cube surrounding the scene
    points = numpy.array([[extents[0, 0], extents[0, 1], extents[0, 2]],
                          [extents[0, 0], extents[0, 1], extents[1, 2]],
                          [extents[0, 0], extents[1, 1], extents[0, 2]],
                          [extents[0, 0], extents[1, 1], extents[1, 2]],
                          [extents[1, 0], extents[0, 1], extents[0, 2]],
                          [extents[1, 0], extents[0, 1], extents[1, 2]],
                          [extents[1, 0], extents[1, 1], extents[0, 2]],
                          [extents[1, 0], extents[1, 1], extents[1, 2]]])

    # find the center of the box
    center = (extents[0, :] + extents[1, :]) / 2
    points = points - center

    # determine the extent of the scene box in the up direction
    up_projection = numpy.dot(points, up)
    height = (1 + margin) * numpy.max(numpy.abs(up_projection)) * 2

    # determine the extent of the scene box in the view direction
    view_projection = numpy.dot(points, v)
    view_distance = numpy.max(view_projection) * 1.10

    # build the camera
    return orthographic(position=center + view_distance * v,
                        look_at=center,
                        up=up,
                        height=height)
