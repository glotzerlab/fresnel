Spheres
-------

.. image:: sphere-hires.png
    :width: 690px
    :alt: Spheres

Spheres example script:

* Geometry: :py:class:`geometry.Sphere <fresnel.geometry.Sphere>`: *radius* = 0.5, *outline_width* = 0.1

    * :py:class:`material <fresnel.material.Material>`: *roughness* = 0.8, *specular* = 0.2
    * :py:class:`outline_material <fresnel.material.Material>`: *solid* = 1, *color* = (0,0,0)

* Lighting: :py:meth:`light.cloudy <fresnel.light.cloudy>`
* Rendered with: :py:class:`tracer.Path <fresnel.tracer.Path>`

.. rubric:: Source code

.. literalinclude:: sphere.py
    :lines: 3-21

.. rubric:: Author

*Joshua A. Anderson*
