Protomer
---------

.. image:: protomer-hires.png
    :width: 690px
    :alt: Protomer

Protomer on the cover of `Nature Chemistry volume 11, issue 3 <https://www.nature.com/nchem/volumes/11/issues/3>`_:

* Ribbon geometry: :py:class:`geometry.Mesh <fresnel.geometry.Mesh>`

    * :py:class:`material <fresnel.material.Material>`: *roughness* = 1.0, *specular* = 1.0, *metal* = 0, *spec_trans* = 0
    * Generated with: `ribbon <https://github.com/fogleman/ribbon>`_

* Molecular surface: :py:class:`geometry.Mesh <fresnel.geometry.Mesh>`

    * :py:class:`material <fresnel.material.Material>`: *roughness* = 2.0, *specular* = 0.95, *metal* = 0, *spec_trans* = 0.95
    * Generated with `MSMS <https://mgl.scripps.edu/people/sanner/html/msms_home.html>`_

* Lighting: :py:meth:`light.lightbox <fresnel.light.lightbox>` with background light
* Rendered with: :py:class:`tracer.Path <fresnel.tracer.Path>`: *samples* = 64, *light_samples* = 32 on the GPU

.. rubric:: Author

*Jens Glaser*
