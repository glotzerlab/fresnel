# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Lights.

"""

from __future__ import division
import math
import collections
import math

from . import _common

class _light_proxy(object):
    def __init__(self, light_list, idx):
        self._light_list = light_list;
        self._idx = idx;

    @property
    def direction(self):
        v = self._light_list._lights.getDirection(self._idx);
        d = (v.x, v.y, v.z)
        return d;

    @direction.setter
    def direction(self, v):
        self._light_list._lights.setDirection(self._idx, _common.vec3f(*v))

    @property
    def color(self):
        v = self._light_list._lights.getColor(self._idx);
        c = (v.r, v.g, v.b);
        return c

    @color.setter
    def color(self, v):
        self._light_list._lights.setColor(self._idx, _common.RGBf(*v))

class LightList(object):
    R""" Manage a list of lights.

    A :py:class:`LightList` stores up to 4 directional lights.
    The maximum limit is partly for performance and partly to enforce good lighting design.

    Each light has a direction and color. The color also defines the intensity of the light, and is defined in
    linear sRGB space (see :py:func:`fresnel.color.linear`).

    You can query the length of a :py:class:`LightList` with :py:func:`len`, append lights to the list with
    :py:meth:`append`, remove all lights with :py:meth:`clear`, and use indexing to query and modify lights
    in the list.

    Each light has the following attributes. These attributes are passed as keyword arguments to :py:meth:`append`
    and are read-write properties of entries accessed by index.

        * direction: 3-tuple defining the direction pointing to the light in camera space.
        * color: A 3-tuple that defines the color and intensity of the light as a linear sRGB value (see :py:func:`fresnel.color.linear`)

    Direction vectors may have any non-zero length, but only the direction the vector points matters. Fresnel will
    normalize the direction vector internally. TODO: Determine how intensity works with area lights before documenting
    it.

    Warning:

        The *_lights* argument is for internal use only.

    .. code-block:: python

        >>> l = LightList()
        >>> print(len(l))
        0
        >>> l.append(Light(direction=(1,0,0), color=(1,1,1)))
        >>> print(len(l))
        1
        >>> print(l[0]).direction
        (1,0,0)
        >>> l[0].direction = (-1,0,0)
        print(l[0]).direction
        (-1,0,0)

    """

    def __init__(self, _lights=None):
        if _lights is None:
            self._lights = _common.Lights();
            self._lights.N = 0;
        else:
            self._lights = _lights;

    def __len__(self):
        return self._lights.N;

    def clear(self):
        R""" Remove all lights.

        :py:meth:`clear` deletes all lights from the list.
        """

        self._lights.N = 0;

    def append(self, direction, color):
        R""" Add a light.

        :py:meth:`append` adds a light to the list.

        Args:

            direction: A 3-tuple that defines the direction the light points in camera space.
            color:  A 3-tuple that defines the color and intensity of the light as a linear sRGB value (see :py:func:`fresnel.color.linear`)
        """

        if len(self) >= 4:
            raise IndexError("Cannot add more than 4 lights")

        i = self._lights.N
        self._lights.setDirection(i, _common.vec3f(*direction))
        self._lights.setColor(i, _common.RGBf(*color))
        self._lights.N = i + 1;

    def __getitem__(self, idx):
        if idx >= self._lights.N:
            raise IndexError("Indexing past the end of the list")

        return _light_proxy(self, idx);

def butterfly():
    R""" Create a butterfly lighting setup.

    The butterfly portrait lighting setup is front lighting with the key light (0) placed high above the camera
    and the fill light (1) below the camera.

    Returns:

        :py:class:`LightList`.
    """

    res = LightList();
    theta1 = 50*math.pi/180;
    res.append(direction=(0, math.sin(theta1), math.cos(theta1)), color=(0.97,0.97,0.97));
    theta2 = -30*math.pi/180;
    res.append(direction=(0, math.sin(theta2), math.cos(theta2)), color=(0.1,0.1,0.1));
    return res

def loop(side='right'):
    R""" Create a loop lighting setup.

    The loop portrait lighting setup places the key light slightly to one side of the camera and slightly up.
    The fill light is on the other side of the camera at the level of the camera.

    Args:

        side (str): 'right' or 'left' to choose which side of the camera to place the key light.

    Returns:

        :py:class:`LightList`.
    """

    sign = {'right': 1, 'left': -1}

    res = LightList();
    phi1 = sign[side]*25*math.pi/180;
    theta1 = (90-20)*math.pi/180;
    res.append(direction=(math.sin(theta1)*math.sin(phi1), math.cos(theta1), math.sin(theta1)*math.cos(phi1)),
               color=(0.95,0.95,0.95));
    phi1 = -sign[side]*40*math.pi/180;
    theta1 = (90)*math.pi/180;
    res.append(direction=(math.sin(theta1)*math.sin(phi1), math.cos(theta1), math.sin(theta1)*math.cos(phi1)),
               color=(0.1,0.1,0.1));
    return res

def rembrandt(side='right'):
    R""" Create a Rembrandt lighting setup.

    The Rembrandt portrait lighting setup places the key light 45 degrees to one side of the camera and slightly up.
    The fill light is on the other side of the camera at the level of the camera.

    Args:

        side (str): 'right' or 'left' to choose which side of the camera to place the key light.

    Returns:

        :py:class:`LightList`.
    """

    sign = {'right': 1, 'left': -1}

    res = LightList();
    phi1 = sign[side]*45*math.pi/180;
    theta1 = (90-20)*math.pi/180;
    res.append(direction=(math.sin(theta1)*math.sin(phi1), math.cos(theta1), math.sin(theta1)*math.cos(phi1)),
               color=(0.99,0.99,0.99));
    phi1 = -sign[side]*45*math.pi/180;
    theta1 = (90)*math.pi/180;
    res.append(direction=(math.sin(theta1)*math.sin(phi1), math.cos(theta1), math.sin(theta1)*math.cos(phi1)),
               color=(0.1,0.1,0.1));
    return res


## TODO: add more types of setups enabled by area lights: ring, side, cloudy_day, raking, ....
