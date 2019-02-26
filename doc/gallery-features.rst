Features
=========

A gallery of **fresnel** renders that demonstrate specific features.

Cuboids
-------

.. image:: gallery/cuboid-hires.png
    :width: 690px
    :alt: Cuboids

Cuboid example script. Utilizes the :py:class:`convex polyhedron geometry <fresnel.geometry.ConvexPolyhedron>` with
metallic outlines, :py:class:`smooth materials <fresnel.material.Material>`,
:py:meth:`lightbox lighting <fresnel.light.lightbox>`, and :py:class:`path tracing <fresnel.tracer.Path>`.
Author: *Joshua A. Anderson*

.. literalinclude:: gallery/cuboid.py
    :lines: 3-29

Spheres
-------

.. image:: gallery/sphere-hires.png
    :width: 690px
    :alt: Spheres

Spheres example script. Utilizes the :py:class:`sphere geometry <fresnel.geometry.Sphere>` with black outlines,
:py:class:`rough materials <fresnel.material.Material>`, :py:meth:`cloudy lighting <fresnel.light.cloudy>`,
and :py:class:`path tracing <fresnel.tracer.Path>`.
Author: *Joshua A. Anderson*

.. literalinclude:: gallery/sphere.py
    :lines: 3-21
