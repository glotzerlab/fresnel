# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""Interactive Qt widgets."""

import sys
import numpy

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
import rowan

from fresnel import tracer, camera

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


class _CameraController3D:
    """Helper class to control camera movement."""

    def __init__(self, camera):
        self.camera = camera
        if isinstance(self.camera, str):
            raise RuntimeError("Cannot control an auto camera")

    def orbit(self, yaw=0, pitch=0, roll=0, factor=-0.0025, slight=False):
        """Orbit the camera about the look_at point."""
        if slight:
            factor = factor * 0.1

        basis = numpy.array(self.camera.basis)

        q1 = rowan.from_axis_angle(basis[1, :], factor * yaw)
        q2 = rowan.from_axis_angle(basis[0, :], factor * pitch)
        q3 = rowan.from_axis_angle(basis[2, :], factor * roll)
        q = rowan.multiply(q2, rowan.multiply(q1, q3))

        v = self.camera.position - self.camera.look_at
        v = rowan.rotate(q, v)

        self.camera.position = self.camera.look_at + v
        self.camera.up = rowan.rotate(q, basis[1, :])

    def pan(self, x, y, slight=False):
        """Pan the camera parallel to the focal plane."""
        # TODO: this should be the height at the focal plane
        factor = self.camera.height

        if slight:
            factor = factor * 0.1

        basis = numpy.array(self.camera.basis)

        delta = factor * (x * basis[0, :] + y * basis[1, :])

        self.camera.position += delta
        self.camera.look_at += delta

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

    """Qt Signal sent when rendering starts at a new camera position."""
    rendering = QtCore.Signal(camera.Camera)

    def __init__(self, scene, max_samples=2000):
        super().__init__()
        self.setWindowTitle("fresnel: scene viewer")

        self.setMinimumSize(10, 10)

        self._scene = scene
        self._max_samples = max_samples

        # fire off a timer to repaint the window as often as possible
        self._repaint_timer = QtCore.QTimer(self)
        self._repaint_timer.timeout.connect(self.update)

        # initialize a single-shot timer to delay resizing
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._resize_done)

        # initialize the tracer
        self._tracer = tracer.Path(device=scene.device, w=10, h=10)
        self._is_rendering = False
        self._initial_resize = True

        # flag to notify view rotation
        self.camera_update_mode = None
        self.mouse_initial_pos = None

        self.camera_controller = _CameraController3D(self._scene.camera)
        self.ipython_display_formatter = 'text'

    def minimumSizeHint(self):  # noqa
        """Specify the minimum window size hint to Qt.

        :meta private:"""
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
        """Paint the window.

        :meta private:"""
        if self._is_rendering:
            # Render the scene
            self._tracer.render(self._scene)

            self._samples += 1
            if self._samples >= self._max_samples:
                self._stop_rendering()

        # Display
        image_array = self._tracer.output

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
        """Adjust the size of the tracer as the window resizes.

        :meta private:"""
        # for the initial window size, resize immediately
        if self._initial_resize:
            self._resize_done()
            self._initial_resize = False
        else:
            # otherwise, defer resizing the tracer until the window sits still
            # for a bit
            self._resize_timer.start(100)

    def _resize_done(self):
        """Resize the tracer after a delay."""
        # resize the tracer
        self._tracer.resize(w=self.width(), h=self.height())
        self._start_rendering()

    def _stop_rendering(self):
        """Stop sampling the scene."""
        self._repaint_timer.stop()
        self._is_rendering = False

    def _start_rendering(self):
        """Start sampling the scene."""
        # send signal
        self.rendering.emit(self._scene.camera)

        self._is_rendering = True
        self._samples = 0
        self._tracer.reset()
        self._repaint_timer.start()

    def mouseMoveEvent(self, event):  # noqa
        """Respond to mouse move events.

        :meta private:"""
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
        """Respond to mouse press events.

        :meta private:"""
        self.mouse_initial_pos = event.pos()
        event.accept()

        if event.button() == QtCore.Qt.LeftButton:
            self.camera_update_mode = 'pitch/yaw'
        elif event.button() == QtCore.Qt.RightButton:
            self.camera_update_mode = 'roll'
        elif event.button() == QtCore.Qt.MiddleButton:
            self.camera_update_mode = 'pan'

    def mouseReleaseEvent(self, event):  # noqa
        """Respond to mouse release events.

        :meta private:"""
        if self.camera_update_mode is not None:
            self.camera_update_mode = None
            event.accept()

    def wheelEvent(self, event):  # noqa
        """Respond to mouse wheel events.

        :meta private:"""
        self.camera_controller.zoom(event.angleDelta().y(),
                                    slight=event.modifiers()
                                    & QtCore.Qt.ControlModifier)
        self._start_rendering()
        event.accept()
