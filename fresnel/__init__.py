# Copyright (c) 2016-2024 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""The fresnel ray tracing package."""

import os
import numpy

from . import geometry  # noqa: F401 - ignore unused import
from . import tracer
from . import camera
from . import color  # noqa: F401 - ignore unused import (users will use)
from . import light
from . import version  # noqa: F401 - ignore unused import (users will use)

from . import _common
if _common.cpu_built():
    from . import _cpu
if _common.gpu_built():
    from . import _gpu


class Device(object):
    """Hardware device to use for ray tracing.

    Args:
        mode (str): Specify execution mode: Valid values are ``auto``, ``gpu``,
            and ``cpu``.
        n (int): Specify the number of cpu threads / GPUs this device will use.
            *None* will use all available threads / devices.

    `Device` defines hardware device to use for ray tracing. `Scene` and
    `Tracer` instances must be attached to a `Device`. You may attach any number
    of scenes and tracers to a single `Device`.

    See Also:
        Tutorials:

        - :doc:`examples/02-Advanced-topics/01-Devices`
        - :doc:`examples/02-Advanced-topics/02-Tracer-methods`

    When mode is ``auto``, the default, `Device` will select GPU rendering if
    available and fall back on CPU rendering if not. Set mode to ``gpu`` or
    ``cpu`` to force a specific mode.

    Important:
        By default (``n==None``), this device will use all available GPUs or CPU
        cores. Set *n* to the number of GPUs or CPU cores this device should
        use. When selecting *n* GPUs, the device selects the first *n* in the
        `available_gpus` list.

    Tip:
        Use only a single `Device` to reduce memory consumption.

    The static member `available_modes` lists which modes are available. For a
    mode to be available, the corresponding module must be enabled at compile
    time. Additionally, there must be at least one GPU present for the ``gpu``
    mode to be available.

    .. code-block:: python

        >>> fresnel.Device.available_modes
        ['gpu', 'cpu', 'auto']
    """

    available_modes = []
    """list[str]: Available execution modes."""

    available_gpus = []
    """list[str]: Available GPUS."""

    def __init__(self, mode='auto', n=None):
        # determine the number of available GPUs
        num_gpus = 0
        if _common.gpu_built():
            num_gpus = _gpu.get_num_available_devices()

        # determine the selected mode
        selected_mode = ''

        if mode == 'auto':
            if num_gpus > 0:
                selected_mode = 'gpu'
            else:
                selected_mode = 'cpu'
                if not _common.cpu_built():
                    raise RuntimeError("No GPUs available AND CPU "
                                       "implementation is not compiled")

        if mode == 'gpu':
            if not _common.gpu_built():
                raise RuntimeError("GPU implementation is not compiled")
            if num_gpus == 0:
                raise RuntimeError("No GPUs are available")
            selected_mode = 'gpu'

        if mode == 'cpu':
            if not _common.cpu_built():
                raise RuntimeError("CPU implementation is not compiled")
            selected_mode = 'cpu'

        if n is None:
            thread_limit = -1
        else:
            thread_limit = int(n)

        # initialize the device
        if selected_mode == 'gpu':
            self.module = _gpu
            self._device = _gpu.Device(
                os.path.dirname(os.path.realpath(__file__)), thread_limit)
            self._mode = 'gpu'
        elif selected_mode == 'cpu':
            self.module = _cpu
            self._device = _cpu.Device(thread_limit)
            self._mode = 'cpu'
        else:
            raise ValueError("Invalid mode")

    @property
    def mode(self):
        """str: The active mode."""
        return self._mode

    def __str__(self):
        """Human readable `Device` summary."""
        return '<fresnel.Device: ' + self._device.describe() + '>'


# determine available Device modes
if _common.gpu_built():
    if _gpu.get_num_available_devices() > 0:
        Device.available_modes.append('gpu')

if _common.cpu_built():
    Device.available_modes.append('cpu')

if len(Device.available_modes) > 0:
    Device.available_modes.append('auto')

# determine available Device GPUs
if _common.gpu_built():
    gpus_str = _gpu.Device.getAllGPUs()
    gpus_list = gpus_str.split('\n')
    if len(gpus_list) >= 2:
        Device.available_gpus = gpus_list[:-1]


class Scene(object):
    """Content of the scene to ray trace.

    Args:
        device (Device): Device to use when rendering the scene.

        camera (camera.Camera): Camera to view the scene. When `None`,
          defaults to::

            camera.Orthographic(position=(0, 0, 100),
                                look_at=(0, 0, 0),
                                up=(0, 1, 0),
                                height=100)

        lights (list[Light]): Lights to light the scene. When `None`, defaults
          to: ``light.rembrandt()``

    `Scene` defines the contents of the scene to be traced, including any number
    of `Geometry` objects, the `Camera`, the `background_color`,
    `background_alpha`, and `lights`.

    Every `Scene` must be associated with a `Device`. For convenience, `Scene`
    creates a default `Device` when *device* is ``None``.

    See Also:
        Tutorials:

        - :doc:`examples/00-Basic-tutorials/00-Introduction`
        - :doc:`examples/00-Basic-tutorials/04-Scene-properties`
        - :doc:`examples/00-Basic-tutorials/05-Lighting-setups`
        - :doc:`examples/02-Advanced-topics/01-Devices`
    """

    def __init__(self, device=None, camera=None, lights=None):
        if device is None:
            device = Device()

        self._device = device
        self._scene = self.device.module.Scene(self.device._device)
        self.geometry = []
        if camera is None:
            self.camera = globals()['camera'].Orthographic(position=(0, 0, 100),
                                                           look_at=(0, 0, 0),
                                                           up=(0, 1, 0),
                                                           height=100)
        else:
            self.camera = camera

        if lights is None:
            self.lights = light.rembrandt()
        else:
            self.lights = lights

        self._tracer = None

    def get_extents(self):
        """Get the extents of the scene.

        Returns:
            (3,2) `numpy.ndarray` of ``numpy.float32``: The lower left and\
                upper right corners of the scene.
        """
        if len(self.geometry) == 0:
            return numpy.array([[0, 0, 0], [0, 0, 0]], dtype=numpy.float32)

        scene_extents = self.geometry[0].get_extents()
        for geom in self.geometry[1:]:
            extents = geom.get_extents()
            scene_extents[0, :] = numpy.min(
                [scene_extents[0, :], extents[0, :]], axis=0)
            scene_extents[1, :] = numpy.max(
                [scene_extents[1, :], extents[1, :]], axis=0)

        return scene_extents

    @property
    def device(self):
        """Device: Device this `Scene` is attached to."""
        return self._device

    @property
    def camera(self):
        """camera.Camera: Camera view parameters."""
        return camera._from_cpp(self._scene.getCamera())

    @camera.setter
    def camera(self, value):
        if isinstance(value, camera.Camera):
            self._scene.setCamera(value._camera)
        else:
            raise TypeError(f"camera {value} is not a fresnel.camera.Camera")

    @property
    def background_color(self):
        """((3, ) `numpy.ndarray` of ``numpy.float32``): Background color \
          linear RGB.

        Note:
            Use `fresnel.color.linear` to convert standard sRGB colors into the
            linear color space used by fresnel.
        """
        c = self._scene.getBackgroundColor()
        return numpy.array([c.r, c.g, c.b], dtype=numpy.float32)

    @background_color.setter
    def background_color(self, value):
        self._scene.setBackgroundColor(_common.RGBf(*value))

    @property
    def background_alpha(self):
        """float: Background alpha (opacity) in the range [0,1]."""
        return self._scene.getBackgroundAlpha()

    @background_alpha.setter
    def background_alpha(self, value):
        self._scene.setBackgroundAlpha(value)

    @property
    def lights(self):
        """list[Light]: Lights in the scene.

        `lights` is a sequence of up to 4 directional lights that apply to the
        scene. Each light has a direction, color, and size.
        """
        return light._LightListProxy(self._scene.getLights())

    @lights.setter
    def lights(self, values):
        tmp = light._LightListProxy()
        for v in values:
            tmp.append(v)

        self._scene.setLights(tmp._lights)


def preview(scene, w=600, h=370, anti_alias=True):
    """Preview a scene.

    Args:
        scene (`Scene`): Scene to render.
        w (int): Output image width (in pixels).
        h (int): Output image height (in pixels).
        anti_alias (bool): Whether to perform anti-aliasing.

    :py:func:`preview` is a shortcut that renders output with `tracer.Preview`.
    """
    t = tracer.Preview(scene.device, w=w, h=h, anti_alias=anti_alias)
    return t.render(scene)


def pathtrace(scene, w=600, h=370, samples=64, light_samples=1):
    """Path trace a scene.

    Args:
        scene (`Scene`): Scene to render.
        w (int): Output image width (in pixels).
        h (int): Output image height (in pixels).
        samples (int): Number of times to sample the pixels of the scene.

        light_samples (int): Number of light samples to take for each pixel
            sample.

    :py:func:`pathtrace` is a shortcut that renders output with `tracer.Path`.
    """
    t = tracer.Path(scene.device, w=w, h=h)
    t.sample(scene, samples=samples, light_samples=light_samples)
    return t.output
