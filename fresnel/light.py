# Copyright (c) 2016-2021 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Lights.

Light objects in a `Scene`.

See Also:
    Tutorials:

    - :doc:`examples/00-Basic-tutorials/04-Scene-properties`
    - :doc:`examples/00-Basic-tutorials/05-Lighting-setups`
"""

import math

from . import _common


class Light(object):
    """A light.

    Args:
        direction ((3, ) `numpy.ndarray` of ``float32``):
            Vector direction the light points (in `Camera` space).

        color ((3, ) `numpy.ndarray` of ``float32``):
            Linear RGB color and intensity of the light.

        theta (float): Half angle of the cone that defines the area of the light
            (in radians).

    In fresnel, lights are area lights at an infinite distance away in the
    given *direction* and are circular with the size set by *theta*. *color*
    sets the light intensity. A ``(0.5, 0.5, 0.5)`` light is twice as bright as
    ``(0.25, 0.25, 0.25)``. Lights are normalized so that ``color = (1, 1, 1)``
    should provide approximately a correct exposure. Color values greater than
    1 are allowed.

    Note:
        *direction* is in `Camera` space. A direction of (1, 0, 0) sets a light
        coming from the right in the image, regardless of the camera
        position.
    """

    def __init__(self, direction, color=(1, 1, 1), theta=0.375):
        self.direction = tuple(direction)
        self.color = tuple(color)
        self.theta = float(theta)

    def __str__(self):
        """Human readable representation of Light."""
        return "<Light object with direction {0} and color {1}>".format(
            self.direction, self.color)


class _LightProxy(object):

    def __init__(self, light_list, idx):
        self._light_list = light_list
        self._idx = idx

    @property
    def direction(self):
        v = self._light_list._lights.getDirection(self._idx)
        d = (v.x, v.y, v.z)
        return d

    @direction.setter
    def direction(self, v):
        self._light_list._lights.setDirection(self._idx, _common.vec3f(*v))

    @property
    def color(self):
        v = self._light_list._lights.getColor(self._idx)
        c = (v.r, v.g, v.b)
        return c

    @color.setter
    def color(self, v):
        self._light_list._lights.setColor(self._idx, _common.RGBf(*v))

    @property
    def theta(self):
        return self._light_list._lights.getTheta(self._idx)

    @theta.setter
    def theta(self, v):
        self._light_list._lights.setTheta(self._idx, v)

    def __str__(self):
        return "<Light proxy object with direction {0} and color {1}>".format(
            self.direction, self.color)


class _LightListProxy(object):

    def __init__(self, _lights=None):
        if _lights is None:
            self._lights = _common.Lights()
            self._lights.N = 0
        else:
            self._lights = _lights

    def __len__(self):
        return self._lights.N

    def clear(self):
        self._lights.N = 0

    def append(self, light):
        if len(self) >= 4:
            raise IndexError("Cannot add more than 4 lights")

        i = self._lights.N
        self._lights.setDirection(i, _common.vec3f(*light.direction))
        self._lights.setColor(i, _common.RGBf(*light.color))
        self._lights.setTheta(i, light.theta)
        self._lights.N = i + 1

    def __getitem__(self, idx):
        if idx >= self._lights.N:
            raise IndexError("Indexing past the end of the list")

        return _LightProxy(self, idx)


def butterfly():
    """Create a butterfly lighting setup.

    The butterfly portrait lighting setup is front lighting with the key light
    (index 0) placed high above the camera and the fill light (index 1) below
    the camera.

    Returns:
        list[Light]: The lights.
    """
    res = []
    theta1 = 50 * math.pi / 180
    res.append(
        Light(direction=(0, math.sin(theta1), math.cos(theta1)),
              color=(1.0, 1.0, 1.0),
              theta=math.pi / 4))
    theta2 = -30 * math.pi / 180
    res.append(
        Light(direction=(0, math.sin(theta2), math.cos(theta2)),
              color=(0.1, 0.1, 0.1),
              theta=math.pi / 2))
    return res


def loop(side='right'):
    """Create a loop lighting setup.

    The loop portrait lighting setup places the key light slightly to one side
    of the camera and slightly up (index 0). The fill light is on the other side
    of the camera at the level of the camera (index 1).

    Args:
        side (str): 'right' or 'left' to choose which side of the camera to
            place the key light.

    Returns:
        list[Light]: The lights.
    """
    sign = {'right': 1, 'left': -1}

    res = []
    phi1 = sign[side] * 25 * math.pi / 180
    theta1 = (90 - 20) * math.pi / 180
    res.append(
        Light(direction=(math.sin(theta1) * math.sin(phi1), math.cos(theta1),
                         math.sin(theta1) * math.cos(phi1)),
              color=(1.0, 1.0, 1.0),
              theta=math.pi / 4))
    phi1 = -sign[side] * 40 * math.pi / 180
    theta1 = (90) * math.pi / 180
    res.append(
        Light(direction=(math.sin(theta1) * math.sin(phi1), math.cos(theta1),
                         math.sin(theta1) * math.cos(phi1)),
              color=(0.1, 0.1, 0.1),
              theta=math.pi / 2))
    return res


def rembrandt(side='right'):
    """Create a Rembrandt lighting setup.

    The Rembrandt portrait lighting setup places the key light  45 degrees to
    one side of the camera and slightly up (index 0). The fill light is on the
    other side of the camera at the level of the camera (index 1).

    Args:
        side (str): 'right' or 'left' to choose which side of the camera to
            place the key light.

    Returns:
        list[Light]: The lights.
    """
    sign = {'right': 1, 'left': -1}

    res = []
    phi1 = sign[side] * 45 * math.pi / 180
    theta1 = (90 - 20) * math.pi / 180
    res.append(
        Light(direction=(math.sin(theta1) * math.sin(phi1), math.cos(theta1),
                         math.sin(theta1) * math.cos(phi1)),
              color=(1.0, 1.0, 1.0),
              theta=math.pi / 4))
    phi1 = -sign[side] * 45 * math.pi / 180
    theta1 = (90) * math.pi / 180
    res.append(
        Light(direction=(math.sin(theta1) * math.sin(phi1), math.cos(theta1),
                         math.sin(theta1) * math.cos(phi1)),
              color=(0.1, 0.1, 0.1),
              theta=math.pi / 2))
    return res


def lightbox():
    """Create a light box lighting setup.

    The light box lighting setup places a single area light that covers the top,
    bottom, left, and right. Use `Path` tracing for best results with this
    setup.

    Returns:
        list[Light]: The lights.
    """
    res = []
    res.append(Light(direction=(0, 0, 1), color=(0.9, 0.9, 0.9), theta=math.pi))
    return res


def cloudy():
    """Create a cloudy day lighting setup.

    The cloudy lighting setup mimics a cloudy day. A strong light comes from all
    directions above. A weaker light comes from all directions below (accounting
    for light reflected off the ground). Use `Path` tracing for best results
    with this setup.

    Returns:
        list[Light]: The lights.
    """
    res = []
    res.append(
        Light(direction=(0, 1, 0), color=(0.9, 0.9, 0.9), theta=math.pi / 2))
    res.append(
        Light(direction=(0, -1, 0), color=(0.1, 0.1, 0.1), theta=math.pi / 2))
    return res


def ring():
    """Create a ring lighting setup.

    The ring lighting setup provides a strong front area light. This type of
    lighting is common in fashion photography. Use `Path` tracing for best
    results with this setup.

    Returns:
        list[Light]: The lights.
    """
    res = []
    res.append(
        Light(direction=(0, 0, 1), color=(0.9, 0.9, 0.9), theta=math.pi / 4))
    return res
