# Copyright (c) 2016-2019 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Ray tracers process a :py:class:`fresnel.Scene` and render output images.

Fresnel provides a :py:class:`Preview` tracer to generate a quick approximate render
and :py:class:`Path` which provides soft shadows, reflections, and other effects.

.. seealso::
    :doc:`examples/000-Introduction`
        Tutorial: Introduction to tracers

    :doc:`examples/202-Tracer-methods`
        Tutorial: Configuring tracer parameters.

"""

import numpy
import math
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
        seed (int): Random number seed.

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
        """

        scene._prepare();
        self._tracer.render(scene._scene);
        return self.output;

    def enable_highlight_warning(self, color=(1,0,1)):
        R""" Enable highlight clipping warnings.

        When a pixel in the rendered image is too bright to represent, make that pixel the given *color* to flag
        the problem to the user.

        Args:

            color (tuple): Color to make the highlight warnings.
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
        the first column contains the lightness histogram and the next 3 contain R,B, and G channel histograms
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

    @property
    def seed(self):
        return self._tracer.getSeed()

    @seed.setter
    def seed(self, value):
        self._tracer.setSeed(value);


class Preview(Tracer):
    R""" Preview ray tracer.

    Args:

        device (:py:class:`Device <fresnel.Device>`): Device to use for rendering.
        w (int): Output image width.
        h (int): Output image height.
        aa_level (int): Amount of anti-aliasing to perform

    Attributes:

        aa_level (int): Amount of anti-aliasing to perform

    .. rubric:: Overview

    The :py:class:`Preview` tracer produces a preview of the scene quickly. It approximates the effect of light
    on materials. The output of the :py:class:`Preview` tracer will look very similar to that from the :py:class:`Path`
    tracer, but will miss soft shadows, reflection, transmittance, and other lighting effects.

    TODO: show examples

    .. rubric:: Anti-aliasing

    Set :py:attr:`aa_level` to control the amount of anti-aliasing performed. The default value of 0 performs
    no anti-aliasing to enable the fastest possible preview renders. A value of 1 samples 2x2 subpixels, a value of 2
    samples 4x4 subpixels, a value of 3 samples 8x8 subpixels, etc ... Samples are jittered with random numbers.
    Different :py:attr:`seed <Tracer.seed>` values will result in different output images.

    TODO: show examples

    .. tip::

        Use :py:attr:`aa_level` = 3 when using the :py:class:`Preview` tracer to render production quality output.
    """

    def __init__(self, device, w, h, aa_level=0):
        self.device = device;
        self._tracer = device.module.TracerDirect(device._device, w, h);

        self.aa_level = aa_level;

    @property
    def aa_level(self):
        aa_n = self._tracer.getAntialiasingN();
        return int(math.log2(aa_n));

    @aa_level.setter
    def aa_level(self, value):
        aa_n = 2**(int(value))
        self._tracer.setAntialiasingN(aa_n);

class Path(Tracer):
    R""" Path tracer.

    Args:

        device (:py:class:`Device <fresnel.Device>`): Device to use for rendering.
        w (int): Output image width.
        h (int): Output image height.

    The path tracer applies advanced lighting effects, including soft shadows, reflections, etc....
    It operates by Monte Carlo sampling. Each call to :py:meth:`render() <Tracer.render()>` performs one sample per pixel.
    The output image is the mean of all the samples. Many samples are required to produce a smooth image.

    :py:meth:`sample()` provides a convenience API to make many samples with a single call.
    """

    def __init__(self, device, w, h):
        self.device = device;
        self._tracer = device.module.TracerPath(device._device, w, h, 1);

    def reset(self):
        R"""
        Clear the output buffer and start sampling a new image. Increment the random number seed so that the
        new image is statistically independent from the previous.
        """

        self._tracer.reset();

    def sample(self, scene, samples, reset=True, light_samples=1):
        R"""
        Args:

            scene (:py:class:`Scene <fresnel.Scene>`): The scene to render.
            samples (int): The number of samples to take per pixel.
            reset (bool): When True, call :py:meth:`reset()` before sampling

        Returns:
            A reference to the current output buffer as a :py:class:`fresnel.util.image_array`.

        Note:
            When *reset* is False, subsequent calls to :py:meth:`sample()` will continue to add samples
            to the current output image. Use the same number of light samples when sampling an image
            in this way.
        """

        if reset:
            self.reset()

        self._tracer.setLightSamples(light_samples);

        for i in range(samples):
            out = self.render(scene);

        # reset the number of light samples to 1 to avoid side effects with future calls to render() by the user
        self._tracer.setLightSamples(1);

        return out;
