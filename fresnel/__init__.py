# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
The fresnel ray tracing package.
"""

from . import geometry
from . import tracer
from . import camera

class Device:
    R""" Hardware device to use for ray tracing.

    :py:class:`Device` defines hardware device to use for ray tracing. :py:class:`Scene` and
    :py:mod:`tracer <fresnel.tracer>` instances must be attached to a :py:class:`Device`. You may attach any number of
    instances to a single :py:class:`Device`.

    .. tip::
        Use only a single :py:class:`Device` to reduce memory consumption.
    """

    def __init__(self):
        try:
            from fresnel import _cpu;
        except ImportError:
            _cpu = None;

        self.module = _cpu;

        self._device = self.module.Device();

class Scene:
    R""" Content of the scene to ray trace.

    Args:

        device (:py:class:`Device`): Device to create this Scene on.

    :py:class:`Scene` defines the contents of the scene to be ray traced, including any number of
    :py:mod:`geometry <fresnel.geometry>` and :py:mod:`light <fresnel.light>` objects.

    Every :py:class:`Scene` attaches to a :py:class:`Device`. For convenience, :py:class:`Scene` creates a default
    :py:class:`Device` when **device** is *None*. If you want a non-default device, you must create it explicitly.

    Attributes:

        device (:py:class:`Device`): Device this Scene is attached to.
    """

    def __init__(self, device=None):
        if device is None:
            device = Device();

        self.device = device;
        self._scene = self.device.module.Scene(self.device._device);
