# Copyright (c) 2016 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Cameras.
"""

class Orthographic:
    R""" Orthographic camera

    Args:
        device (:py:class:`Device <fresnel.Device>`): Device to use for rendering.
        position (tuple): $\vec{p}$, the position of the camera.
        look_at (tuple): $\vec{l}$, the point the camera points at.
        up (tuple): $\vec{u}$, the vector pointing up.
        height: $h$, the height of the screen in world distance units.

    :py:class:`Orthographic` defines the parameters of an orthographic camera.

    An orthographic camera is defined by a center position $\vec{p}$, directions $\vec{d}$, $\vec{u}$, and $\vec{r}$,
    and a height $h$. For convenience, this API accepts a look at position $\vec{l}. Then, $\vec{d}$ is the normal
    vector that points from $\vec{p}$ to $\vec{l}$.

    The vectors $\vec{d}$, $\vec{u}$, and $\vec{r}$ form a right handed orthonormal coordinate system. The vector
    $\vec{u}$ points "up" on the screen and $\vec{r}$ points to the right. For convenience, you may provide vectors of
    any length in the approximate desired direction. On construction, :py:class:`Orthographic` will normalize vectors
    and form an orthonormal set. The up vector need not be perpendicular to $\vec{d}$, but it must not be parallel.

    In a standard coordinate system, imagine the screen with the x direction pointing right ($\vec{r}$), the
    y direction pointing up ($\vec{u}$), and z pointing *out* of the screen ($-\vec{d}$). With such a camera at 0,0,0,
    only objects at negative z would be visible.

    """
    def __init__(self, device, position, look_at, up, height):
        self.device = device;
        u = (look_at[0] - position[0], look_at[1] - position[1], look_at[2] - position[2]);

        vec3f = self.device.module.vec3f;
        self._camera = self.device.module.Camera(vec3f(*position), vec3f(*u), vec3f(*up), height);
