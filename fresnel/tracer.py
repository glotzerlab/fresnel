# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Ray tracers.
"""

import numpy
from . import camera

class Tracer:
    R""" Base class for all ray tracers.

    :py:class:`Tracer` provides operations common to all ray tracer classes.

    Each :py:class:`Tracer` instance stores a pixel output buffer. When you :py:meth:`render` a
    :py:class:`Scene <fresnel.Scene>`, the current data stored in the buffer is overwritten with the new image.

    TODO: Return output buffer as its own buffer object directly. Users can use it with jupyter, pillow, numpy,
    etc... as desired. Don't force a specific format on the user.

    Note:

        You cannot instantiate a Tracer directly. Use one of the sub classes.
    """
    def __init__(self):
        raise RuntimeError("Use a specific tracer class");

    def resize(self, w, h):
        R""" Resize the output buffer.

        Args:

            w (int): New output buffer width.
            h (int): New output buffer height.

        Warning:
            :py:meth:`resize` clears any existing image in the output buffer.
        """

        self._tracer.resize(w, h);

    def render(self, scene):
        R""" Render a scene.

        Args:

            scene (:py:class:`Scene <fresnel.Scene>`): The scene to render.

        Returns:
            A numpy array pointing to the output buffer.

        Render the given scene and write the resulting pixels into the output buffer.

        Warning:
            :py:meth:`render` clears any existing image in the output buffer.
        """

        self._tracer.render(scene._scene);
        a = numpy.array(self._tracer.getOutputBuffer(), copy=False);
        return(numpy.uint8(a*255));

    def set_camera(self, camera):
        R""" Set the camera.

        Args:
            camera (:py:mod:`fresnel.camera`): Camera to set.

        Set a new camera view. The next time the scene is rendered (:py:meth:`render`), it will be from this
        new view.
        """
        self._tracer.setCamera(camera._camera);

class Whitted(Tracer):
    R""" Whitted ray tracer.

    Args:

        device (:py:class:`Device <fresnel.Device>`): Device to use for rendering.
        w (int): Output buffer width.
        h (int): Output buffer height.

    The Whitted ray tracer is the most basic type of ray tracer. It traces a single ray per pixel. The color of the
    pixel depends on the geometry the ray hits, its material, and the lights in the :py:class:`Scene <fresnel.Scene>`.
    Because of its simplicity, the Whitted tracer is extremely fast.

    :py:class:`Whitted` supports:

    * Directional lights
    * Materials
    * Cel shading
    """

    def __init__(self, device, w, h, camera=camera.Orthographic(position=(0,0, 1), look_at=(0,0,0), up=(0,1,0), height=3)):
        self.device = device;
        self._tracer = device.module.TracerWhitted(device._device, w, h);
        self.set_camera(camera);
