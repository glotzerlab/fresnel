# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Lights.

"""

import math
import collections

from . import _common

Light = collections.namedtuple('Light', ['direction', 'color']);
Light.__doc__ = """Light(direction, color)

Directional light

Attributes:

    direction: A 3-tuple that defines the direction the light points.
    color:  A 3-tuple that defines the color and intensity of the light as a linear sRGB value (see :py:func:`fresnel.color.linear`)
"""

class LightList:
    R""" Manage a list of lights.

    A :py:class:`LightList` stores up to 4 directional lights to light a :py:class:`Scene <fresnel.Scene>`.
    The maximum limit is partly for performance and partly to enforce good lighting design.

    Each light has a direction and color. The color also defines the intensity of the light, and is defined in
    linear sRGB space (see :py:func:`fresnel.color.linear`).

    You can query the length of a :py:class:`LightList` with :py:func:`len`, append lights to the list with
    :py:meth:`append`, remove all lights with :py:meth:`clear`, and use indexing to query and modify lights
    in the list. :py:meth:`append` and indexed access work with :py:class:`Light` data structures.

    Direction vectors may have any non-zero length, but only the direction the vector points matters. Fresnel will
    normalize the direction vector internally. TODO: Determine how intensity works with area lights before documenting
    it.

    .. code-block:: ruby

        l = LightList()
        print(len(l))           # 0
        l.append(Light(direction=(1,0,0), color=(1,1,1)))
        l[0].direction = (0,1,0)
        print(l[0].direction)   # (1,0,0)

    """

    def __init__(self, lights=None):
        if lights is None:
            self._lights = _common.Lights;
        else:
            self._lights = lights;

    def __len__(self):
        return self._lights.N;

    def clear(self):
        R""" Remove all lights.

        :py:meth:`clear` deletes all lights from the list.
        """

        self._lights.N = 0;

    def append(self, light):
        R""" Add a light.

        :py:meth:`append` adds a light to the list.

        Args:

            light (Light): The light to add
        """

        if len(self) >= 4:
            raise IndexError("Cannot add more than 4 lights")

        i = self._lights.N
        self._lights.setDirection(i, _common.vec3f(*light.direction))
        self._lights.setColor(i, _common.RGBf(*light.color))
        self._lights.N = i + 1;

    def __getitem__(self, idx):
        if idx >= self._lights.N:
            raise IndexError("Indexing past the end of the list")

        v = self._lights.getDirection(idx);
        d = (v.x, v.y, v.z)
        v = self._lights.getColor(idx);
        c = (v.r, v.g, v.b);
        return Light(direction=d, color=c);
