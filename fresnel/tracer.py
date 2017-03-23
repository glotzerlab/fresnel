# Copyright (c) 2016-2017 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Ray tracers.
"""

import numpy
from . import camera
from . import util
from . import _common

class Tracer(object):
    R""" Base class for all ray tracers.

    :py:class:`Tracer` provides operations common to all ray tracer classes.

    Each :py:class:`Tracer` instance stores a pixel output buffer. When you :py:meth:`render` a
    :py:class:`Scene <fresnel.Scene>`, the current data stored in the buffer is overwritten with the new image.

    Note:

        You cannot instantiate a Tracer directly. Use one of the sub classes.

    Attributes:

        output (:py:class:`fresnel.util.image_array`): Reference to the current output buffer (modified by :py:meth:`render`)
        linear_output (:py:class:`fresnel.util.array`): Reference to the current output buffer in linear color space (modified by :py:meth:`render`)
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
            A reference to the current output buffer as a :py:class:`fresnel.util.image_array`.

        Render the given scene and write the resulting pixels into the output buffer.

        Warning:
            :py:meth:`render` clears any existing image in the output buffer.
        """

        scene._prepare();
        self._tracer.render(scene._scene);
        return self.output;

    def enable_highlight_warning(self, color=(1,0,1)):
        R""" Enable highlight clipping warnings.

        When a pixel in the rendered image is too bright to represent, make that pixel the given *color* to flag
        the problem to the user.

        Args:

            color (tuple[float]): Color to make the highlight warnings.
        """
        self._tracer.enableHighlightWarning(_common.RGBf(*color));

    def disable_highlight_warning(self):
        R""" Disable the highlight clipping warnings.
        """
        self._tracer.disableHighlightWarning();

    def histogram(self):
        R""" Compute a histogram of the image.

        The histogram is computed as a lightness in the sRGB color space. The histogram is computed only over the
        visible pixels in the image, fully transparent pixels are ignored. The returned histogram is nbins x 4,
        the first column contains the lightness histogram and the next 3 contain R,B,G channel histograms
        respectively.

        Return:

            (histogram, bin_positions).
        """

        a = numpy.array(self.linear_output[:])
        img_sel = a[:,:,3] > 0
        img = a[img_sel]
        r = img[:,0]
        g = img[:,1]
        b = img[:,2]
        l = 0.21*r + 0.72*g + 0.07*b
        gamma_l = l**(1/2.2);

        n=512;
        l_hist, bins = numpy.histogram(gamma_l, bins=n, range=[0,1]);
        r_hist, bins = numpy.histogram(r**(1/2.2), bins=n, range=[0,1]);
        g_hist, bins = numpy.histogram(g**(1/2.2), bins=n, range=[0,1]);
        b_hist, bins = numpy.histogram(b**(1/2.2), bins=n, range=[0,1]);

        out = numpy.stack((l_hist, r_hist, g_hist, b_hist), axis=1)

        return out, bins[1:]

    @property
    def output(self):
        return util.image_array(self._tracer.getSRGBOutputBuffer(), geom=None)

    @property
    def linear_output(self):
        return util.array(self._tracer.getLinearOutputBuffer(), geom=None)

class Direct(Tracer):
    R""" Direct ray tracer.

    Args:

        device (:py:class:`Device <fresnel.Device>`): Device to use for rendering.
        w (int): Output image width.
        h (int): Output image height.

    The Direct ray tracer a basic ray tracer. It traces a single ray per pixel. The color of the
    pixel depends on the geometry the ray hits, its material, and the lights in the :py:class:`Scene <fresnel.Scene>`.
    Because of its simplicity, the Direct tracer is extremely fast.

    :py:class:`Direct` supports:

    * Directional lights
    * Materials
    """

    def __init__(self, device, w, h):
        self.device = device;
        self._tracer = device.module.TracerDirect(device._device, w, h);
