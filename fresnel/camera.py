# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""
Cameras.
"""

import numpy
import math

from . import _common


class Camera(object):
    """Camera.

    A `Camera` defines the view into the `Scene`.

    `Camera` space is a coordinate system centered on the camera's position.
    Positive *x* points to the right in the image, positive *y* points up, and
    positive *z* points out of the screen. `Camera` space shares units with
    `Scene` space.

    `Camera` provides common methods and properties for all camera
    implementations. `Camera` cannot be used directly, use one of the
    subclasses.

    See:
        * `Orthographic`
        * `Perspective`
    """

    def __init__(self, _camera):
        # The Python level `Camera` class keeps a reference to a C++
        # `UserCamera` class up to date with all parameter changes.
        self._camera = _camera

    @property
    def position(self):
        """(`numpy.ndarray` or `array_like`): (``3`` : ``float32``):
                The position of the camera.
        """
        return numpy.array([self._camera.position.x,
                self._camera.position.y,
                self._camera.position.z], dtype=numpy.float32)

    @position.setter
    def position(self, value):
        if len(value) != 3:
            raise ValueError("position must have length 3")
        self._camera.position = _common.vec3f(*value)

    @property
    def look_at(self):
        """(`numpy.ndarray` or `array_like`): (``3`` : ``float32``):
                The point the camera looks at.

        `position` - `look_at` defines the +z direction in camera space.
        """
        return numpy.array([self._camera.look_at.x,
                self._camera.look_at.y,
                self._camera.look_at.z], dtype=numpy.float32)

    @look_at.setter
    def look_at(self, value):
        if len(value) != 3:
            raise ValueError("look_at must have length 3")
        self._camera.look_at = _common.vec3f(*value)

    @property
    def up(self):
        """(`numpy.ndarray` or `array_like`): (``3`` : ``float32``):
                A vector that points up.

        `up` defines the +y direction in camera space.
        """
        return (self._camera.up.x, self._camera.up.y, self._camera.up.z)

    @up.setter
    def up(self, value):
        if len(value) != 3:
            raise ValueError("up must have length 3")
        self._camera.up = _common.vec3f(*value)

    @property
    def height(self):
        """float: The height of the image plane.
        """
        return self._camera.h

    @height.setter
    def height(self, value):
        self._camera.h = float(value)

    @property
    def basis(self):
        """(`numpy.ndarray` or `array_like`): (``3x3`` : ``float32``):
                Orthonormal camera basis.

        `basis` is computed from `position`, `look_at`, and `up`. The 3 vectors
        of the basis define the +x, +y, and +z camera space directions in
        scene space.
        """
        b = _common.CameraBasis(self._camera)
        return numpy.array([(b.u.x, b.u.y, b.u.z),
                (b.v.x, b.v.y, b.v.z),
                (b.w.x, b.w.y, b.w.z)], dtype=numpy.float32)


class Orthographic(Camera):
    """Orthographic camera.

    Args:
        position (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - the
            position of the camera.
        look_at (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - the
            point the camera looks at (the center of the focal plane).
        up (`numpy.ndarray` or `array_like`): (``3`` : ``float32``) - a vector
            pointing up.
        height (float): the height of the image plane.

    An orthographic camera traces parallel rays from the image plane into the
    scene. Lines that are parallel in the `Scene` will remain parallel in the
    rendered image.

    `position` is the center of the image plane in `Scene` space. `look_at` is
    the point in `Scene` space that will be in the center of the image.
    Together, these vectors define the image plane which is perpendicular to the
    line from `position` to `look_at`. Objects in front of the plane will appear
    in the rendered image, objects behind the plane will not.

    `up` is a vector in `Scene` space that defines which direction points up (+y
    direction in the camera space). `up` does not need to be perpendicular to
    the line from *position* to *look_at*, but it must not be parallel to that
    line. `height` sets the height of the image in `Scene` units. The image
    width is determined by the aspect ratio of the image.
    """

    def __init__(self, position, look_at, up, height):
        cam = _common.UserCamera()
        cam.model = _common.CameraModel.orthographic

        super().__init__(cam)

        self.position = position
        self.look_at = look_at
        self.up = up
        self.height = height

    def __repr__(self):
        s = "fresnel.camera.Orthographic("
        s += f"position={self.position}, "
        s += f"look_at={self.look_at}, "
        s += f"up={self.up}, "
        s += f"height={self.height})"
        return s

    @classmethod
    def fit(cls, scene, view='auto', margin=0.05):
        """Fit a camera to a `Scene`

        Create a camera that fits the entire height of the scene in the image
        plane.

        Args:
            scene (`Scene`): Fit the camera to this scene.
            view (str): Select view
            margin (float): Fraction of extra space to leave on the top and
                bottom of the scene.

        *view* may be 'auto', 'isometric', or 'front'.

        The isometric view is an orthographic projection from a particular angle
        so that the x,y, and z directions are equal lengths. The front view is
        an orthographic projection where +x points to the right, +y points up
        and +z points out of the screen in the image plane. 'auto' automatically
        selects 'isometric' for 3D scenes and 'front' for 2D scenes.
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
            raise ValueError('The camera cannot be fit because the scene has no'
                             ' geometries. Add geometries to the scene before'
                             ' calling fit.')

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
        return cls(position=center + view_distance * v,
                            look_at=center,
                            up=up,
                            height=height)


class Perspective(Camera):
    """Perspective camera.

    TODO:
    """

    def __init__(self,
                 position,
                 look_at,
                 up,
                 focus_distance=10,
                 focal_length=.5,
                 f_stop=math.inf,
                 height=0.24):
        cam = _common.UserCamera()
        cam.model = _common.CameraModel.perspective

        super().__init__(cam)

        self.position = position
        self.look_at = look_at
        self.up = up
        self.focus_distance = focus_distance
        self.focal_length = focal_length
        self.f_stop = f_stop
        self.height = height

    def __repr__(self):
        s = "fresnel.camera.Perspective("
        s += f"position={self.position}, "
        s += f"look_at={self.look_at}, "
        s += f"up={self.up}, "
        s += f"focus_distance={self.focus_distance}, "
        s += f"focal_length={self.focal_length}, "
        s += f"f_stop={self.f_stop}, "
        s += f"height={self.height})"
        return s

    @property
    def focal_length(self):
        """Focal length of the camera lens.

        The focal length relative to the image `height` sets the field of view.
        Given a fixed `height`, a larger `focal_length` gives a narrower field
        of view.

        Tip:
            With the default height of 0.24, typical focal lengths range from
            .18 (wide angle) to 0.5 (normal) to 6.0 (telephoto).

        See:
            `vertical_field_of_view`
        """
        return self._camera.f

    @focal_length.setter
    def focal_length(self, value):
        self._camera.f = value

    @property
    def f_stop(self):
        """F-stop ratio for the lens.

        Set the aperture of the opening into the lens in f-stops. This sets the
        range of the scene that is in sharp focus. Smaller values of `f_stop`
        result in more background blur.

        Tip:
            Use `depth_of_field` to set the range of sharp focus in `Scene`
            distance units.
        """
        return self._camera.f_stop

    @f_stop.setter
    def f_stop(self, value):
        self._camera.f_stop = value

    @property
    def focus_distance(self):
        """Distance to the focal plane.

        The focus distance is the distance from the camera position to
        the center of focal plane.

        Tip:
            Use `focus_on` to compute the focus distance to a particular point
            in the `Scene`.
        """
        return self._camera.focus_distance

    @focus_distance.setter
    def focus_distance(self, value):
        self._camera.focus_distance = value

    @property
    def depth_of_field(self):
        """The distance about the focal plane in sharp focus.

        The area of sharp focus extends in front and behind the focal plane. The
        distance between the front and back areas of sharp focus is the depth
        of field.

        The depth of field is a function of `focus_distance`, `focal_length`,
        `f_stop`, and `height`.

        Setting `depth_of_field` computes `f_stop` to obtain the desired depth
        of field as a function of `focus_distance`, `focal_length`, and
        `height`.

        Note:
            `depth_of_field` does not remain fixed after setting it.
        """

        f = self.focal_length
        N = self.f_stop
        s = self.focus_distance

        # c is the circle of confusion. A commonly accepted value for 35 mm
        # cameras is 0.03mm. 35mm film is 24mm high, so c = h/(0.03 mm / 24 mm)
        # => c = h/800
        c = self.height / 800

        H = f**2 / (N * c)

        # the depth of field is infinite when focusing past the hyperfocal
        # distance
        if H - (s - f) <= 0 or H + (s - f) <= 0:
            return math.inf

        return s * H * (1 / (H - (s - f)) - 1 / (H + (s - f)))

    @depth_of_field.setter
    def depth_of_field(self, value):
        f = self.focal_length
        d = value
        s = self.focus_distance

        # c is the circle of confusion. A commonly accepted value for 35 mm
        # cameras is 0.03mm. 35mm film is 24mm high, so c = h/(0.03 mm / 24 mm)
        # => c = h/800
        c = self.height / 800

        N = (math.sqrt(c**2 * f**4 * (d**2 + s**2) * (f - s)**2)
            + c * f**2 * s * (f - s))/(c**2 * d * (f - s)**2)
        self.f_stop = N

    @property
    def focus_on(self):
        """(`numpy.ndarray` or `array_like`): (``3`` : ``float32``):
                A point in the focal plane.

        The area of sharp focus extends in front and behind the focal plane.

        The focal plane is a function of `focus_distance`, `position`, and
        `look_at`.

        Setting `focus_on` computes `focus_distance` so that the given point
        is on the focal plane.

        Note:
            `focus_on` does not remain fixed after setting it.
        """
        d = -self.basis[2, :]
        return self.position + d * self.focus_distance

    @focus_on.setter
    def focus_on(self, value):
        if len(value) != 3:
            raise ValueError("focus_on must have length 3")

        d = -self.basis[2, :]
        self.focus_distance = numpy.dot(d, value - self.position)

    @property
    def vertical_field_of_view(self):
        """float: Vertical field of view.

        The vertical field of view is the angle (in radians) that the camera
        covers in the +y direction. It is a function of `focal_length`
        and `height`.

        Setting `vertical_field_of_view` computes `focal_length` to achieve
        the given field of view.

        Note:
            `vertical_field_of_view` does not remain fixed after setting it.
        """
        return 2 * math.atan(self.height / (2 * self.focal_length))

    @vertical_field_of_view.setter
    def vertical_field_of_view(self, value):
        self.focal_length = self.height / (2 * math.tan(value/2))


def _from_cpp(cam):
    """Make a Python camera object from a C++ UserCamera.

    There is only one UserCamera class with mode flags at the C++ level while
    we expose them as separate classes at the Python level.
    """
    if cam.model == _common.CameraModel.orthographic:
        result = Orthographic(position=(0,0,0),
                              look_at=(0,0,1),
                              up=(0,1,0),
                              height=1)
        result._camera = cam
    elif cam.model == _common.CameraModel.perspective:
        result = Perspective(position=(0,0,0),
                             look_at=(0,0,1),
                             up=(0,1,0),
                             focus_distance=1)
        result._camera = cam
    else:
        raise RuntimeError("Invalid camera model")

    return result
