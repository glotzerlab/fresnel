# Copyright (c) 2016-2021 The Regents of the University of Michigan
# Part of fresnel, released under the BSD 3-Clause License.

"""Ray tracers process a `Scene` and render output images.

* `Preview` generates a quick approximate render.
* `Path` which provides soft shadows, reflections, and other effects.

See Also:
    Tutorials:

    - :doc:`examples/00-Basic-tutorials/00-Introduction`
    - :doc:`examples/02-Advanced-topics/02-Tracer-methods`
"""

import numpy
from . import util
from . import _common


class Tracer(object):
    """Base class for all ray tracers.

    `Tracer` provides operations common to all ray tracer classes.

    Each `Tracer` instance stores a pixel output buffer. When you `render` a
    `Scene`, the `output` is updated.

    Note:
        You cannot instantiate `Tracer` directly. Use one of the subclasses.
    """

    def __init__(self):
        raise RuntimeError("Use a specific tracer class")

    def resize(self, w, h):
        """Resize the output buffer.

        Args:
            w (int): New output buffer width.
            h (int): New output buffer height.

        Warning:
            `resize` clears the output buffer.
        """
        self._tracer.resize(w, h)

    def render(self, scene):
        """Render a scene.

        Args:
            scene (`Scene <fresnel.Scene>`): The scene to render.

        Returns:
            A reference to the current output buffer as a
            `fresnel.util.ImageArray`.

        Render the given scene and write the resulting pixels into the output
        buffer.
        """
        self._tracer.render(scene._scene)
        return self.output

    def enable_highlight_warning(self, color=(1, 0, 1)):
        """Enable highlight clipping warnings.

        When a pixel in the rendered image is too bright to represent, make that
        pixel the given *color* to flag the problem to the user.

        Args:
            color (tuple): Color to make the highlight warnings.
        """
        self._tracer.enableHighlightWarning(_common.RGBf(*color))

    def disable_highlight_warning(self):
        """Disable the highlight clipping warnings."""
        self._tracer.disableHighlightWarning()

    def histogram(self):
        """Compute a histogram of the image.

        The histogram is computed as a lightness in the sRGB color space. The
        histogram is computed only over the visible pixels in the image, fully
        transparent pixels are ignored. The returned histogram is nbins x 4, the
        first column contains the lightness histogram and the next 3 contain
        R,B, and G channel histograms respectively.

        Return:
            (histogram, bin_positions).
        """
        a = numpy.array(self.linear_output[:])
        img_sel = a[:, :, 3] > 0
        img = a[img_sel]
        r = img[:, 0]
        g = img[:, 1]
        b = img[:, 2]
        l = 0.21 * r + 0.72 * g + 0.07 * b  # noqa: E741 - allow l as a name
        gamma_l = l**(1 / 2.2)

        n = 512
        l_hist, bins = numpy.histogram(gamma_l, bins=n, range=[0, 1])
        r_hist, bins = numpy.histogram(r**(1 / 2.2), bins=n, range=[0, 1])
        g_hist, bins = numpy.histogram(g**(1 / 2.2), bins=n, range=[0, 1])
        b_hist, bins = numpy.histogram(b**(1 / 2.2), bins=n, range=[0, 1])

        out = numpy.stack((l_hist, r_hist, g_hist, b_hist), axis=1)

        return out, bins[1:]

    @property
    def output(self):
        """ImageArray: Reference to the current output buffer.

        Note:
            The output buffer is modified by `render` and `resize`.
        """
        return util.ImageArray(self._tracer.getSRGBOutputBuffer(), geom=None)

    @property
    def linear_output(self):
        """Array: Reference to the current output buffer in linear color space.

        Note:
            The output buffer is modified by `render` and `resize`.
        """
        return util.Array(self._tracer.getLinearOutputBuffer(), geom=None)

    @property
    def seed(self):
        """int: Random number seed."""
        return self._tracer.getSeed()

    @seed.setter
    def seed(self, value):
        self._tracer.setSeed(value)


class Preview(Tracer):
    """Preview ray tracer.

    Args:
        device (`Device`): Device to use.

        w (int): Output image width.

        h (int): Output image height.

        anti_alias (bool): Whether to perform anti-aliasing. If True, uses an
            64 samples.

    .. rubric:: Overview

    The `Preview` tracer produces a preview of the scene quickly. It
    approximates the effect of light on materials. The output of the `Preview`
    tracer will look very similar to that from the `Path` tracer, but will miss
    soft shadows, reflection, transmittance, depth of field and other effects.

    .. rubric:: Anti-aliasing

    The default value of `anti_alias` is True to smooth sharp edges in
    the image. The anti-aliasing level corresponds to ``aa_level=3`` in fresnel
    versions up to 0.11.0. Different `seed` values will result in different
    output images.
    """

    def __init__(self, device, w, h, anti_alias=True):
        self.device = device
        self._tracer = device.module.TracerDirect(device._device, w, h)
        self.anti_alias = anti_alias

    @property
    def anti_alias(self):
        """bool: Whether to perform anti-aliasing."""
        return self._tracer.getAntialiasingN() > 1

    @anti_alias.setter
    def anti_alias(self, value):
        if value:
            self._tracer.setAntialiasingN(8)
        else:
            self._tracer.setAntialiasingN(1)


class Path(Tracer):
    """Path tracer.

    Args:
        device (`Device`): Device to use.
        w (int): Output image width.
        h (int): Output image height.

    The path tracer applies advanced lighting effects, including soft shadows,
    reflections, and depth of field. It operates by Monte Carlo sampling. Each
    call to `render` performs one sample per pixel. The `output` image is the
    mean of all the samples. Many samples are required to produce a smooth
    image. `sample` provides a convenience API to make many samples with a
    single call.
    """

    def __init__(self, device, w, h):
        self.device = device
        self._tracer = device.module.TracerPath(device._device, w, h, 1)

    def reset(self):
        """Clear the output buffer.

        Start sampling a new image. Increment the random number seed so that the
        new image is statistically independent from the previous.
        """
        self._tracer.reset()

    def sample(self, scene, samples, reset=True, light_samples=1):
        r"""Sample the image.

        Args:
            scene (`Scene`): The scene to render.

            samples (int): The number of samples to take per pixel.

            reset (bool): When True, call `reset` before sampling

            light_samples (int): The number of light samples per primary camera
                ray.

        As an unbiased renderer, the sampling noise will scale as
        :math:`\frac{1}{\sqrt{\text{total_samples}}}`, where ``total_samples``
        is ``samples*light_samples``.

        The ``samples`` parameter controls the number of samples from the camera
        (depth of field and antialiasing). ``light_samples`` is the number of
        rays shot from the first intersection of the primary camera ray.

        Using ``(samples=N, light_samples=1)`` would have an equal number of
        camera and lighting samples and would produce an excellent image. Using
        ``(samples=N // M, light_samples=M)`` (where ``M`` is some integer
        10-100) will similarly produce a similarly good image, provided ``N >>
        M``, and may have better performance. On the GPU, using ``light_samples
        > 1`` can boost performance moderately. On the CPU, it can boost
        performance slightly due to improved cache coherency.

        Returns:
            ImageArray: A reference to the current `output` buffer.

        Note:
            When *reset* is ``False``, subsequent calls to `sample` will
            continue to add samples to the current output image. Use the same
            number of light samples when sampling an image in this way.
        """
        if reset:
            self.reset()

        self._tracer.setLightSamples(light_samples)

        for i in range(samples):
            out = self.render(scene)

        # reset the number of light samples to 1 to avoid side effects with
        # future calls to render() by the user
        self._tracer.setLightSamples(1)

        return out
