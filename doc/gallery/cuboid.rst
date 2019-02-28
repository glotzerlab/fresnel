Cuboids
-------

.. image:: cuboid-hires.png
    :width: 690px
    :alt: Cuboids

Cuboid example script:

* Geometry: :py:class:`geometry.ConvexPolyhedron <fresnel.geometry.ConvexPolyhedron>`: *outline_width* = 0.015

    * :py:class:`material <fresnel.material.Material>`: *roughness* = 0.1, *specular* = 1
    * :py:class:`outline_material <fresnel.material.Material>`: *roughness* = 0.1, *metal* = 1, *color* = (0.95,0.93,0.88)

* Lighting: :py:meth:`light.lightbox <fresnel.light.lightbox>`
* Rendered with: :py:class:`tracer.Path <fresnel.tracer.Path>`

.. rubric:: Source code

.. literalinclude:: cuboid.py
    :lines: 3-29

.. rubric:: Author

*Joshua A. Anderson*
