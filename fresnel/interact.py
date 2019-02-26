# Copyright (c) 2016-2019 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause License.

R"""
Interactive Qt widgets.
"""

import sys

# workaround bug in ipython that prevents pyside2 importing
# https://github.com/jupyter/qtconsole/pull/280
try:
    import IPython.external.qt_loaders
    if type(sys.meta_path[0]) == IPython.external.qt_loaders.ImportDenier:
        del sys.meta_path[0]
except:
    pass

from PySide2 import QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore
import numpy
import time
import collections
import math

from . import tracer, camera

# initialize QApplication
# but not in sphinx
if 'sphinx' not in sys.modules:
    app = QtCore.QCoreApplication.instance();
    if app is None:
        app = QtWidgets.QApplication(['fresnel'])

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
    x, y, z = v
    theta /= 2
    w = math.cos(theta)
    x = x * math.sin(theta)
    y = y * math.sin(theta)
    z = z * math.sin(theta)
    return w, x, y, z

class CameraController3D:
    def __init__(self, camera):
        self.camera = camera;
        if type(self.camera) == type('str'):
            raise RuntimeError("Cannot control an auto camera");

    def orbit(self, yaw=0, pitch=0, roll=0, factor=-0.0025, slight=False):
        if slight:
            factor = factor * 0.1;

        r, d, u = self.camera.basis

        q1 = axisangle_to_q(u, factor * yaw)
        q2 = axisangle_to_q(r, factor * pitch)
        q3 = axisangle_to_q(d, factor * roll)
        q = q_mult(q1, q2);
        q = q_mult(q, q3);

        px, py, pz = self.camera.position
        ax, ay, az = self.camera.look_at
        v = (px - ax, py - ay, pz - az)
        vx, vy, vz = qv_mult(q, v)

        self.camera.position = (vx + ax, vy + ay, vz + az)
        self.camera.up = qv_mult(q, u)

    def pan(self, x, y, slight=False):
        # TODO: this should be the height at the focal plane
        factor = self.camera.height

        if slight:
            factor = factor * 0.1;

        r, d, u = self.camera.basis

        rx, ry, rz = r
        ux, uy, uz = u
        delta_x, delta_y, delta_z = factor*(x*rx + y*ux), factor*(x*ry + y*uy), factor*(x*rz + y*uz)

        px, py, pz = self.camera.position
        ax, ay, az = self.camera.look_at

        self.camera.position = px+delta_x, py+delta_y, pz+delta_z
        self.camera.look_at = ax+delta_x, ay+delta_y, az+delta_z

    def zoom(self, s, slight=False):
        R""" Zoom the view in or out
        """
        factor = 0.0015

        if slight:
            factor = factor * 0.1;

        # TODO: different types of zoom for perspective cameras
        self.camera.height = self.camera.height * (1 - s*factor)

class SceneView(QtWidgets.QWidget):
    R""" View a fresnel Scene in real time

    :py:class:`SceneView` is a PySide2 widget that displays a :py:class:`fresnel.Scene`, rendering it with
    :py:class:`fresnel.tracer.Path` interactively. Use the mouse to rotate the camera view.

    Args:

        scene (:py:class:`Scene <fresnel.Scene>`): The scene to display.
        max_samples (int): Sample until ``max_samples`` have been averaged.

    * Left click to pitch and yaw
    * Right click to roll
    * Middle click to pan
    * Hold ctrl to make small adjustments

    .. rubric:: Using in a standalone script

    To use SceneView in a standalone script, import the :py:mod:`fresnel.interact` module, create your :py:class:`fresnel.Scene`, instantiate the
    :py:class:`SceneView`, show it, and start the app event loop.

    .. code-block:: python

        import fresnel, fresnel.interact
        # build Scene
        view = fresnel.interact.SceneView(scene)
        view.show()
        fresnel.interact.app.exec_();

    .. rubric:: Using with jupyter notebooks

    To use SceneView in a jupyter notebook, import PySide2.QtCore and activate jupyter's qt5 integration.

    .. code-block:: python

        from PySide2 import QtCore
        % gui qt


    Import the :py:mod:`fresnel.interact` module, create your :py:class:`fresnel.Scene`, and instantiate the
    :py:class:`SceneView`. Do not call the app event loop, jupyter is already running the event loop in the background.
    When the SceneView object is the result of a cell, it will automatically show and activate focus.

    .. code-block:: python

        import fresnel, fresnel.interact
        # build Scene
        fresnel.interact.SceneView(scene)

    Note:

        The interactive window will open on the system that *hosts* jupyter.

    .. seealso::
        :doc:`examples/02-Advanced-topics/03-Interactive-scene-view`
            Tutorial: Interactive scene display

    """
    def __init__(self, scene, max_samples=2000):
        super().__init__()
        self.setWindowTitle("fresnel: scene viewer")

        self.setMinimumSize(10,10)

        self.max_samples = max_samples;

        # pick a default camera if one isn't already set
        self.scene = scene
        if type(self.scene.camera) == type('str'):
            self.scene.camera = camera.fit(self.scene);

        # fire off a timer to repaint the window as often as possible
        self.repaint_timer = QtCore.QTimer(self)
        self.repaint_timer.timeout.connect(self.update)

        # initialize a single-shot timer to delay resizing
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.resize_done)

        # initialize the tracer
        self.tracer = tracer.Path(device=scene.device, w=10, h=10)
        self.rendering = False
        self.initial_resize = True

        # flag to notify view rotation
        self.camera_update_mode = None;
        self.mouse_initial_pos = None;

        self.camera_controller = CameraController3D(self.scene.camera)

    def _repr_html_(self):
        self.show();
        self.raise_();
        self.activateWindow();
        return "<p><i>scene view opened in a new window...</i></p>"

    def minimumSizeHint(self):
        return QtCore.QSize(1610, 1000)

    def setScene(self, scene):
        R""" Set a new scene

        Args:

            scene (:py:class:`Scene <fresnel.Scene>`): The scene to render.

        Also call setScene when you make any changes to the scene so that SceneView window will re-render the scene
        with the changes.
        """
        self.scene = scene;
        self.start_rendering()

    def paintEvent(self, event):
        if self.rendering:
            # Render the scene
            self.tracer.render(self.scene)

            self.samples += 1;
            if self.samples >= self.max_samples:
                self.stop_rendering()

        # Display
        image_array = self.tracer.output;

        # display the rendered scene in the widget
        image_array.buf.map();
        img = QtGui.QImage(image_array.buf,image_array.shape[1],image_array.shape[0],QtGui.QImage.Format_RGBA8888)
        qp = QtGui.QPainter(self)
        target = QtCore.QRectF(0, 0, self.width(), self.height());
        source = QtCore.QRectF(0.0, 0.0, image_array.shape[1], image_array.shape[0]);
        #qp.drawImage(0,0,img);
        qp.drawImage(target, img, source);
        qp.end()
        image_array.buf.unmap();

    def resizeEvent(self, event):
        delta = event.size() - event.oldSize();
        r = max(delta.width() / event.size().width(), delta.height() / event.size().height())

        # for the initial window size, resize immediately
        if self.initial_resize:
            self.resize_done()
            self.initial_resize = False;
        else:
            # otherwise, defer resizing the tracer until the window sits still for a bit
            self.resize_timer.start(300)

    def resize_done(self):
        # resize the tracer
        self.tracer.resize(w=self.width(), h=self.height());
        self.start_rendering()

    def stop_rendering(self):
        self.repaint_timer.stop()
        self.rendering = False;

    def start_rendering(self):
        self.rendering = True;
        self.samples = 0;
        self.tracer.reset()
        self.repaint_timer.start()


    def mouseMoveEvent(self, event):
        delta = event.pos() - self.mouse_initial_pos;
        self.mouse_initial_pos = event.pos();

        if self.camera_update_mode == 'pitch/yaw':
             self.camera_controller.orbit(yaw=delta.x(),
                                          pitch=delta.y(),
                                          slight=event.modifiers() & QtCore.Qt.ControlModifier)

        elif self.camera_update_mode == 'roll':
            self.camera_controller.orbit(roll=delta.x(),
                                         slight=event.modifiers() & QtCore.Qt.ControlModifier)

        elif self.camera_update_mode == 'pan':
            h = self.height()
            self.camera_controller.pan(x=-delta.x()/h,
                                       y=delta.y()/h,
                                       slight=event.modifiers() & QtCore.Qt.ControlModifier)


        self.start_rendering()
        self.update()
        event.accept();


    def mousePressEvent(self, event):
        self.mouse_initial_pos = event.pos()
        event.accept();

        if event.button() == QtCore.Qt.LeftButton:
            self.camera_update_mode = 'pitch/yaw';
        elif event.button() == QtCore.Qt.RightButton:
            self.camera_update_mode = 'roll';
        elif event.button() == QtCore.Qt.MiddleButton:
            self.camera_update_mode = 'pan';

    def mouseReleaseEvent(self, event):
        if self.camera_update_mode is not None:
            self.camera_update_mode = None;
            event.accept()

    def wheelEvent(self, event):
        self.camera_controller.zoom(event.angleDelta().y(),
                                    slight=event.modifiers() & QtCore.Qt.ControlModifier)
        self.start_rendering()
        event.accept()
