# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""Interactive Qt widgets."""

import sys

# workaround bug in ipython that prevents pyside2 importing
# https://github.com/jupyter/qtconsole/pull/280
try:
    import IPython.external.qt_loaders
    if type(sys.meta_path[0]) == IPython.external.qt_loaders.ImportDenier:
        del sys.meta_path[0]
except:  # noqa
    pass

from PySide2 import QtGui
from PySide2 import QtCore
from PySide2 import QtWidgets
import math

from . import tracer, camera

# initialize QApplication
# but not in sphinx
if 'sphinx' not in sys.modules:
    from PySide2.QtWidgets import QWidget

    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(['fresnel'])
else:
    # work around bug where sphinx does not find classes that have QWidget
    # as a parent
    QWidget = object


def _q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def _q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def _qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return _q_mult(_q_mult(q1, q2), _q_conjugate(q1))[1:]


def _axisangle_to_q(v, theta):
    x, y, z = v
    theta /= 2
    w = math.cos(theta)
    x = x * math.sin(theta)
    y = y * math.sin(theta)
    z = z * math.sin(theta)
    return w, x, y, z


class _CameraController3D:

    def __init__(self, camera):
        self.camera = camera
        if isinstance(self.camera, str):
            raise RuntimeError("Cannot control an auto camera")

    def orbit(self, yaw=0, pitch=0, roll=0, factor=-0.0025, slight=False):
        if slight:
            factor = factor * 0.1

        r, d, u = self.camera.basis

        q1 = _axisangle_to_q(u, factor * yaw)
        q2 = _axisangle_to_q(r, factor * pitch)
        q3 = _axisangle_to_q(d, factor * roll)
        q = _q_mult(q1, q2)
        q = _q_mult(q, q3)

        px, py, pz = self.camera.position
        ax, ay, az = self.camera.look_at
        v = (px - ax, py - ay, pz - az)
        vx, vy, vz = _qv_mult(q, v)

        self.camera.position = (vx + ax, vy + ay, vz + az)
        self.camera.up = _qv_mult(q, u)

    def pan(self, x, y, slight=False):
        # TODO: this should be the height at the focal plane
        factor = self.camera.height

        if slight:
            factor = factor * 0.1

        r, d, u = self.camera.basis

        rx, ry, rz = r
        ux, uy, uz = u
        delta_x = factor * (x * rx + y * ux)
        delta_y = factor * (x * ry + y * uy)
        delta_z = factor * (x * rz + y * uz)

        px, py, pz = self.camera.position
        ax, ay, az = self.camera.look_at

        self.camera.position = px + delta_x, py + delta_y, pz + delta_z
        self.camera.look_at = ax + delta_x, ay + delta_y, az + delta_z

    def zoom(self, s, slight=False):
        """Zoom the view."""
        factor = 0.0015

        if slight:
            factor = factor * 0.1

        # TODO: different types of zoom for perspective cameras
        self.camera.height = self.camera.height * (1 - s * factor)


class SceneView(QWidget):
    """View a fresnel Scene in real time.

    `SceneView` is a PySide2 widget that displays a `Scene`, rendering it with
    `Path` interactively. Use the mouse to rotate the camera view.

    Args:
        scene (`Scene`): The scene to display.
        max_samples (int): Sample a total of ``max_samples``.

    * Left click to pitch and yaw
    * Right click to roll
    * Middle click to pan
    * Hold ctrl to make small adjustments

    .. rubric:: Using in a standalone script

    To use SceneView in a standalone script, import the `fresnel.interact`
    module, create a `Scene`, instantiate the `SceneView`, show it, and start
    the app event loop.

    .. code-block:: python

        import fresnel, fresnel.interact
        # build scene
        view = fresnel.interact.SceneView(scene)
        view.show()
        fresnel.interact.app.exec_();

    .. rubric:: Using with Jupyter notebooks

    To use SceneView in a Jupyter notebook, import PySide2.QtCore and activate
    Jupyter's qt5 integration.

    .. code-block:: python

        from PySide2 import QtCore
        % gui qt


    Import the :py:mod:`fresnel.interact` module, create a `Scene`, and
    instantiate the `SceneView`. Do not call the app event loop, Jupyter is
    already running the event loop in the background. When the SceneView object
    is the result of a cell, it will automatically show and activate focus.

    .. code-block:: python

        import fresnel, fresnel.interact
        # build Scene
        fresnel.interact.SceneView(scene)

    Note:
        The interactive window will open on the system that *hosts* Jupyter.

    See Also:
        Tutorials:

        - :doc:`examples/02-Advanced-topics/03-Interactive-scene-view`
    """

    rendering = QtCore.Signal(camera.Camera)

    def __init__(self, scene, max_samples=2000):
        super().__init__()
        self.setWindowTitle("fresnel: scene viewer")

        self.setMinimumSize(10, 10)

        self.max_samples = max_samples

        # pick a default camera if one isn't already set
        self._scene = scene
        if isinstance(self._scene.camera, str):
            self._scene.camera = camera.fit(self._scene)

        # fire off a timer to repaint the window as often as possible
        self.repaint_timer = QtCore.QTimer(self)
        self.repaint_timer.timeout.connect(self.update)

        # initialize a single-shot timer to delay resizing
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._resize_done)

        # initialize the tracer
        self.tracer = tracer.Path(device=scene.device, w=10, h=10)
        self._is_rendering = False
        self.initial_resize = True

        # flag to notify view rotation
        self.camera_update_mode = None
        self.mouse_initial_pos = None

        self.camera_controller = _CameraController3D(self._scene.camera)
        self.ipython_display_formatter = 'text'

    def minimumSizeHint(self):  # noqa
        return QtCore.QSize(1610, 1000)

    @property
    def scene(self):
        """Scene: The scene rendered in this view."""
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene
        self._start_rendering()

    def paintEvent(self, event):  # noqa
        if self._is_rendering:
            # Render the scene
            self.tracer.render(self._scene)

            self.samples += 1
            if self.samples >= self.max_samples:
                self._stop_rendering()

        # Display
        image_array = self.tracer.output

        # display the rendered scene in the widget
        image_array.buf.map()
        img = QtGui.QImage(image_array.buf, image_array.shape[1],
                           image_array.shape[0], QtGui.QImage.Format_RGBA8888)
        qp = QtGui.QPainter(self)
        target = QtCore.QRectF(0, 0, self.width(), self.height())
        source = QtCore.QRectF(0.0, 0.0, image_array.shape[1],
                               image_array.shape[0])

        qp.drawImage(target, img, source)
        qp.end()
        image_array.buf.unmap()

    def resizeEvent(self, event):  # noqa
        # for the initial window size, resize immediately
        if self.initial_resize:
            self._resize_done()
            self.initial_resize = False
        else:
            # otherwise, defer resizing the tracer until the window sits still
            # for a bit
            self.resize_timer.start(300)

    def _resize_done(self):
        # resize the tracer
        self.tracer.resize(w=self.width(), h=self.height())
        self._start_rendering()

    def _stop_rendering(self):
        self.repaint_timer.stop()
        self._is_rendering = False

    def _start_rendering(self):
        # send signal
        self.rendering.emit(self._scene.camera)

        self._is_rendering = True
        self.samples = 0
        self.tracer.reset()
        self.repaint_timer.start()

    def mouseMoveEvent(self, event):  # noqa
        delta = event.pos() - self.mouse_initial_pos
        self.mouse_initial_pos = event.pos()

        if self.camera_update_mode == 'pitch/yaw':
            self.camera_controller.orbit(yaw=delta.x(),
                                         pitch=delta.y(),
                                         slight=event.modifiers()
                                         & QtCore.Qt.ControlModifier)

        elif self.camera_update_mode == 'roll':
            self.camera_controller.orbit(roll=delta.x(),
                                         slight=event.modifiers()
                                         & QtCore.Qt.ControlModifier)

        elif self.camera_update_mode == 'pan':
            h = self.height()
            self.camera_controller.pan(x=-delta.x() / h,
                                       y=delta.y() / h,
                                       slight=event.modifiers()
                                       & QtCore.Qt.ControlModifier)

        self._start_rendering()
        self.update()
        event.accept()

    def mousePressEvent(self, event):  # noqa
        self.mouse_initial_pos = event.pos()
        event.accept()

        if event.button() == QtCore.Qt.LeftButton:
            self.camera_update_mode = 'pitch/yaw'
        elif event.button() == QtCore.Qt.RightButton:
            self.camera_update_mode = 'roll'
        elif event.button() == QtCore.Qt.MiddleButton:
            self.camera_update_mode = 'pan'

    def mouseReleaseEvent(self, event):  # noqa
        if self.camera_update_mode is not None:
            self.camera_update_mode = None
            event.accept()

    def wheelEvent(self, event):  # noqa
        self.camera_controller.zoom(event.angleDelta().y(),
                                    slight=event.modifiers()
                                    & QtCore.Qt.ControlModifier)
        self._start_rendering()
        event.accept()
