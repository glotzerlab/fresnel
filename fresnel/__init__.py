# Copyright (c) 2016-2018 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
The fresnel ray tracing package.

Attributes:
    __version__ (str): Fresnel version

"""

import os
import numpy

from . import geometry
from . import tracer
from . import camera
from . import color
from . import light

from . import _common
if _common.cpu_built():
    from . import _cpu
if _common.gpu_built():
    from .import _gpu

__version__ = "0.6.0"

class Device(object):
    R""" Hardware device to use for ray tracing.

    Args:

        mode (str): Specify execution mode: Valid values are `auto`, `gpu`, and `cpu`.
        n (int): Specify the number of cpu threads / gpus this device will use.
                     *None* sets no limit.

    :py:class:`Device` defines hardware device to use for ray tracing. :py:class:`Scene` and
    :py:mod:`tracer <fresnel.tracer>` instances must be attached to a :py:class:`Device`. You may attach any number of
    scenes and tracers to a single :py:class:`Device`.

    When mode is `auto`, the default, :py:class:`Device` GPU rendering and
    fall back on CPU rendering if there is no GPU available or GPU support was not compiled in. Set mode to
    `gpu` or `cpu` to force a specific mode.

    .. important::
        By default (n==None), this device will use all available GPUs or CPU cores. Set *n* to the number of GPUs or CPU
        cores this device should use. When selecting *n* GPUs, the device selects the first *n* in the
        :py:attr:`available_gpus` list.

    .. tip::
        Use only a single :py:class:`Device` to reduce memory consumption.

    The static member :py:attr:`available_modes` lists which modes are available. For a mode to be available, the
    corresponding module must be enabled at compile time. Additionally, there must be at least one GPU present
    for the ``gpu`` mode to be available.

    .. code-block:: python

        >>> fresnel.Device.available_modes
        ['gpu', 'cpu', 'auto']

    Attributes:

        available_modes (list): List of the available execution modes (static member).
        available_gpus (list): List of the available gpus (static member).
        mode (string): The active mode

    """

    available_modes = []
    available_gpus = []

    def __init__(self, mode='auto', n=None):
        # determine the number of available GPUs
        num_gpus = 0;
        if _common.gpu_built():
            num_gpus = _gpu.get_num_available_devices();

        # determine the selected mode
        selected_mode = '';

        if mode == 'auto':
            if num_gpus > 0:
                selected_mode = 'gpu'
            else:
                selected_mode = 'cpu'
                if not _common.cpu_built():
                    raise RuntimeError("No GPUs available AND CPU implementation is not compiled");

        if mode == 'gpu':
            if not _common.gpu_built():
                raise RuntimeError("GPU implementation is not compiled");
            if num_gpus == 0:
                raise RuntimeError("No GPUs are available");
            selected_mode = 'gpu';

        if mode == 'cpu':
            if not _common.cpu_built():
                raise RuntimeError("CPU implementation is not compiled");
            selected_mode = 'cpu';

        if n is None:
            thread_limit = -1
        else:
            thread_limit = int(n)

        # inititialize the device
        if selected_mode == 'gpu':
            self.module = _gpu;
            self._device = _gpu.Device(os.path.dirname(os.path.realpath(__file__)), thread_limit);
            self.mode = 'gpu'
        elif selected_mode == 'cpu':
            self.module = _cpu;
            self._device = _cpu.Device(thread_limit);
            self.mode = 'cpu'
        else:
            raise ValueError("Invalid mode");

    def __str__(self):
        return '<fresnel.Device: ' + self._device.describe() + '>'

# determine available Device modes
if _common.gpu_built():
    if _gpu.get_num_available_devices() > 0:
        Device.available_modes.append('gpu');

if _common.cpu_built():
    Device.available_modes.append('cpu');

if len(Device.available_modes) > 0:
    Device.available_modes.append('auto');

# determine available Device GPUs
if _common.gpu_built():
    gpus_str = _gpu.Device.getAllGPUs();
    gpus_list = gpus_str.split('\n')
    if len(gpus_list) >= 2:
        Device.available_gpus = gpus_list[:-1]

class Scene(object):
    R""" Content of the scene to ray trace.

    Args:

        device (:py:class:`Device`): Device to create this Scene on.

    :py:class:`Scene` defines the contents of the scene to be ray traced, including any number of
    :py:mod:`geometry <fresnel.geometry>` objects, the :py:mod:`camera <fresnel.camera>`,
    :py:attr:`background color <background_color>`, :py:attr:`background alpha <background_alpha>`,
    and the :py:attr:`lights`.

    Every :py:class:`Scene` attaches to a :py:class:`Device`. For convenience, :py:class:`Scene` creates a default
    :py:class:`Device` when **device** is *None*. If you want a non-default device, you must create it explicitly.

    .. rubric:: Lights

    :py:attr:`lights` is a sequence of up to 4 directional lights that apply to the scene globally. Each light has a
    direction and color. You can assign lights using one of the predefined setups:

    .. code-block:: python

        scene.lights = fresnel.light.butterfly()

    You can assign a sequence of :py:class:`Light <fresnel.light.Light>` objects:

    .. code-block:: python

        scene.lights = [fresnel.light.Light(direction=(1,2,3))]

    You can modify the lights in place:

    .. code-block:: python

        >>> print(len(scene.lights))
        2
        >>> l.append(fresnel.light.Light(direction=(1,0,0), color=(1,1,1)))
        >>> print(len(3))
        1
        >>> print(l[2]).direction
        (1,0,0)
        >>> l[0].direction = (-1,0,0)
        >>> print(l[0]).direction
        (-1,0,0)

    Attributes:

        device (:py:class:`Device`): Device this Scene is attached to.
        camera (:py:class:`camera.Camera`): Camera view parameters, or 'auto' to automatically choose a camera.
        background_color (tuple[float]): Background color (r,g,b) as a tuple or other 3-length python object, in the
                                         linearized color space. Use :py:func:`fresnel.color.linear` to convert standard
                                         sRGB colors
        background_alpha (float): Background alpha (opacity).
        lights (list of `light.Light`): Globals lights in the scene.
    """

    def __init__(self, device=None, camera='auto', lights=light.rembrandt()):
        if device is None:
            device = Device();

        self.device = device;
        self._scene = self.device.module.Scene(self.device._device);
        self.geometry = [];
        self.camera = camera;
        self.lights = lights;
        self._tracer = None;

    def get_extents(self):
        R""" Get the extents of the scene

        Returns:
            [[minimum x, minimum y, minimum z],
             [maximum x, maximum y, maximum z]]
        """
        if len(self.geometry) == 0:
            return numpy.array([[0,0,0],[0,0,0]], dtype=numpy.float32);

        scene_extents = self.geometry[0].get_extents();
        for geom in self.geometry[1:]:
            extents = geom.get_extents();
            scene_extents[0,:] = numpy.min([scene_extents[0,:], extents[0,:]], axis=0)
            scene_extents[1,:] = numpy.max([scene_extents[1,:], extents[1,:]], axis=0)

        return scene_extents;

    @property
    def camera(self):
        if self.auto_camera:
            return 'auto';
        else:
            return camera.Camera(self._scene.getCamera());

    @camera.setter
    def camera(self, value):
        if value == 'auto':
            self.auto_camera = True;
        else:
            self._scene.setCamera(value._camera);
            self.auto_camera = False;

    @property
    def background_color(self):
        color = self._scene.getBackgroundColor();
        return (color.r, color.g, color.b);

    @background_color.setter
    def background_color(self, value):
        self._scene.setBackgroundColor(_common.RGBf(*value));

    @property
    def background_alpha(self):
        return self._scene.getBackgroundAlpha();

    @background_alpha.setter
    def background_alpha(self, value):
        self._scene.setBackgroundAlpha(value);

    @property
    def lights(self):
        return light._lightlist_proxy(self._scene.getLights())

    @lights.setter
    def lights(self, values):
        tmp = light._lightlist_proxy()
        for v in values:
            tmp.append(v);

        self._scene.setLights(tmp._lights);

    def _prepare(self):
        if self.auto_camera:
            cam = camera.fit(self);
            self._scene.setCamera(cam._camera);

def preview(scene, w=600, h=370, aa_level=0):
    R""" Preview a scene.

    Args:

        scene (:py:class:`Scene`): Scene to render.
        w (int): Output image width.
        h (int): Output image height.
        aa_level (int): Amount of anti-aliasing to perform

    :py:func:`preview` is a shortcut to rendering output with the :py:class:`Preview <tracer.Preview>` tracer.
    See the :py:class:`Preview <tracer.Preview>` tracer for a complete description.
    """

    t = tracer.Preview(scene.device, w=w, h=h, aa_level=aa_level);
    return t.render(scene);

def pathtrace(scene, w=600, h=370, samples=64, light_samples=1):
    R""" Path trace a scene.

    Args:

        scene (:py:class:`Scene`): Scene to render.
        w (int): Output image width.
        h (int): Output image height.
        samples (int): Number of times to sample the pixels of the scene.
        light_samples (int): Number of light samples to take for each pixel sample.

    :py:func:`pathtrace` is a shortcut to rendering output with the :py:class:`Path <tracer.Path>` tracer.
    See the :py:class:`Path <tracer.Path>` tracer for a complete description.
    """

    t = tracer.Path(scene.device, w=w, h=h);
    t.sample(scene, samples=samples, light_samples=light_samples)
    return t.output;
