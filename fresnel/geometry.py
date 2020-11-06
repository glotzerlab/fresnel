# Copyright (c) 2016-2020 The Regents of the University of Michigan
# This file is part of the Fresnel project, released under the BSD 3-Clause
# License.

"""Geometric primitives.

Geometry defines objects that are visible in a `Scene`. The base class
`Geometry` provides common operations and properties. Instantiate specific
geometry class to add objects to a `Scene`.

See Also:
    Tutorials:

    - :doc:`examples/00-Basic-tutorials/01-Primitive-properties`
    - :doc:`examples/00-Basic-tutorials/02-Material-properties`
    - :doc:`examples/00-Basic-tutorials/03-Outline-materials`
    - :doc:`examples/02-Advanced-topics/00-Multiple-geometries`
"""

from . import material
from . import util
import numpy


class Geometry(object):
    """Geometry base class.

    `Geometry` provides operations and properties common to all geometry
    classes.

    Note:
        You cannot instantiate a Geometry directly. Use one of the subclasses.
    """

    def __init__(self):
        raise RuntimeError("Use a specific geometry class")

    def enable(self):
        """Enable the geometry.

        When enabled, the geometry will be visible in the `Scene`.

        See Also:
            `disable`
        """
        self._geometry.enable()

    def disable(self):
        """Disable the geometry.

        When disabled, the geometry will not visible in the `Scene`.

        See Also:
            `enable`
        """
        self._geometry.disable()

    def remove(self):
        """Remove the geometry from the scene.

        After calling `remove`, the geometry is no longer part of the scene. It
        cannot be added back into the scene. Use `disable` and `enable` hide
        geometry reversibly.
        """
        self._geometry.remove()
        self.scene.geometry.remove(self)

    @property
    def material(self):
        """Material: Define how light interacts with the geometry."""
        return material._MaterialProxy(self)

    @material.setter
    def material(self, mat):
        self._geometry.setMaterial(mat._get_cpp_material())

    @property
    def outline_material(self):
        """Material: Define how light interacts with the geometry's outline."""
        return material._OutlineMaterialProxy(self)

    @outline_material.setter
    def outline_material(self, mat):
        self._geometry.setOutlineMaterial(mat._get_cpp_material())

    @property
    def outline_width(self):
        """float: Width of the outline in scene units."""
        return self._geometry.getOutlineWidth()

    @outline_width.setter
    def outline_width(self, width):
        self._geometry.setOutlineWidth(width)


class Cylinder(Geometry):
    """Cylinder geometry.

    Define a set of spherocylinder primitives with individual start and end
    positions, radii, and colors.

    Args:
        scene (Scene): Add the geometry to this scene.

        points ((N, 2, 3) `numpy.ndarray` of ``float32``): *N* cylinder start
            and end points.

        radius ((N, ) `numpy.ndarray` of ``float32``): Radius of each cylinder.

        color ((N, 2, 3) `numpy.ndarray` of ``float32``): Color of each start
            and end point.

        N (int): Number of cylinders in the geometry. If ``None``, determine
            *N* from *points*.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/01-Cylinder-geometry`

    Hint:
        Avoid costly memory allocations and type conversions by specifying
        primitive properties in the appropriate array type.

    Tip:
        When all cylinders are the same size or color, pass a single value
        and NumPy will broadcast it to all elements of the array.
    """

    def __init__(self,
                 scene,
                 points=((0, 0, 0), (0, 0, 0)),
                 radius=0.5,
                 color=(0, 0, 0),
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(points)

        self._geometry = scene.device.module.GeometryCylinder(scene._scene, N)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width

        self.points[:] = points
        self.radius[:] = radius
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        A = self.points[:, 0]
        B = self.points[:, 1]
        r = self.radius[:]
        r = r.reshape(len(r), 1)
        res = numpy.array([
            numpy.min([numpy.min(A - r, axis=0),
                       numpy.min(B - r, axis=0)],
                      axis=0),
            numpy.max([numpy.max(A + r, axis=0),
                       numpy.max(B + r, axis=0)],
                      axis=0)
        ])
        return res

    @property
    def points(self):
        """(N, 2, 3) `Array`: The start and end points of the cylinders."""
        return util.Array(self._geometry.getPointsBuffer(), geom=self)

    @property
    def radius(self):
        """(N, ) `Array`: The radii of the cylinders."""
        return util.Array(self._geometry.getRadiusBuffer(), geom=self)

    @property
    def color(self):
        """(N, 2, 3) `Array`: Color of each start and end point."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)


class Box(Cylinder):
    """Box geometry.

    Generate a triclinic box outline with `spherocylinders <Cylinder>`.

    Args:
        scene (Scene): Add the geometry to this scene.

        box ((1, ), (3, ), or (6, ) `numpy.ndarray` of ``float32``): Box
            parameters.

        radius (float): Radius of box edges.

        box_color ((3, ) `numpy.ndarray` of ``float32``): Color of the box
            edges.

    Note:
        A 1-element *box* array expands to a cube. A 3-element *box* array
        ``[Lx, Ly, Lz]`` expands to an orthorhobic cuboid, and a 6-element
        *box* array represents a fully triclinic box in the same format as
        GSD and HOOMD: ``[Lx, Ly, Lz, xy, xz, yz]``.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/05-Box-geometry`
        - :doc:`examples/02-Advanced-topics/05-GSD-visualization`

    Note:
        The Box class is constructed from `spherocylinders <Cylinder>`, which
        can be modified individually. The convenience attributes ``box_radius``
        and ``box_color`` can be used to set the thickness and color of the
        entire box.
    """

    def __init__(self, scene, box, box_radius=0.5, box_color=[0, 0, 0]):

        super().__init__(scene=scene,
                         N=12,
                         material=material.Material(solid=1.0))
        self._box = self._from_box(box)
        self.points[:] = self._generate_points(self._box)

        self.box_radius = box_radius

        self.box_color = box_color

    def _from_box(self, box):
        """Duck type the box from a valid input.

        Boxes can be a number, list, dictionary, or object with attributes.
        """
        try:
            # Handles freud.box.Box and namedtuple
            Lx = box.Lx
            Ly = box.Ly
            Lz = getattr(box, 'Lz', 0)
            xy = getattr(box, 'xy', 0)
            xz = getattr(box, 'xz', 0)
            yz = getattr(box, 'yz', 0)
        except AttributeError:
            try:
                # Handle dictionary-like
                Lx = box['Lx']
                Ly = box['Ly']
                Lz = box.get('Lz', 0)
                xy = box.get('xy', 0)
                xz = box.get('xz', 0)
                yz = box.get('yz', 0)
            except (IndexError, KeyError, TypeError):
                try:
                    if not len(box) in [1, 3, 6]:
                        raise ValueError(
                            "List-like objects must have length 1, 3, or 6 to "
                            "be converted to a box.")
                    # Handle list-like
                    Lx = box[0]
                    Ly = box[0] if len(box) == 1 else box[1]
                    Lz = box[0] if len(box) == 1 else box[2]
                    xy, xz, yz = box[3:6] if len(box) == 6 else (0, 0, 0)
                except TypeError:
                    if isinstance(box, int) or isinstance(box, float):
                        # Handle int or float
                        Lx = box
                        Ly = box
                        Lz = box
                        xy = 0
                        xz = 0
                        yz = 0
                    else:
                        raise TypeError(f"unsupported box type {type(box)}")
        return (Lx, Ly, Lz, xy, xz, yz)

    def _generate_points(self, box):
        """Helper function to take a box and calculate the 12 edges."""
        Lx = box[0]
        Ly = box[1]
        Lz = box[2]
        xy = box[3]
        xz = box[4]
        yz = box[5]

        # Follow hoomd convention
        box_matrix = numpy.array([[Lx, xy * Ly, xz * Lz], [0, Ly, yz * Lz],
                                  [0, 0, Lz]])
        a_1, a_2, a_3 = box_matrix.T
        #           F--------------H
        #          /|             /|
        #         / |            / |
        #        D--+-----------E  |
        #        |  |           |  |
        #        |  |           |  |
        #        |  |           |  |
        #        |  C-----------+--G
        #        | /            | /
        #        |/             |/
        #        A--------------B
        # Translate A so that 0, 0, 0 is the center of the box
        A = -(a_1 + a_2 + a_3) / 2
        B = A + a_1
        C = A + a_2
        D = A + a_3
        E = A + a_1 + a_3
        F = A + a_2 + a_3
        G = A + a_1 + a_2
        H = A + a_1 + a_2 + a_3
        # Define all edges
        box_points = numpy.asarray([
            [A, B],
            [A, C],
            [A, D],
            [B, E],
            [B, G],
            [C, G],
            [C, F],
            [D, E],
            [D, F],
            [E, H],
            [F, H],
            [G, H],
        ])
        return box_points

    @property
    def box(self):
        """(1, ), (3, ), or (6, ) `numpy.ndarray` of ``float32``: Box\
            parameters.

        Set `box` to update the shape of the box.
        """
        return self._box

    @box.setter
    def box(self, box):
        self._box = self._from_box(box)
        self.points[:] = self._generate_points(self._box)

    @property
    def box_color(self):
        """(3, ) `numpy.ndarray` of ``float32``: Color of the box edges.

        Note:
            This property sets the color of the `material <Geometry.material>`.
        """
        return self.material.color

    @box_color.setter
    def box_color(self, color):
        self.material.color = color

    @property
    def box_radius(self):
        """(float): Radius of box edges."""
        return self.radius[:][0]

    @box_radius.setter
    def box_radius(self, radius):
        self.radius[:] = radius


class Polygon(Geometry):
    """Polygon geometry.

    Define a set of simple polygon primitives in the xy plane with individual
    positions, rotation angles, and colors.

    Args:
        scene (Scene): Add the geometry to this scene.

        vertices ((N_vert, 2) `numpy.ndarray` of ``float32``): Polygon vertices.

        position ((N, 2) `numpy.ndarray` of ``float32``): Position of each
            polygon.

        angle ((N, ) `numpy.ndarray` of ``float32``): Orientation angle of each
            polygon (in radians).

        color ((N, 3) `numpy.ndarray` of ``float32``): Color of each polygon.

        rounding_radius (float): Rounding radius for spheropolygons.

        N (int): Number of polygons in the geometry. If ``None``, determine
            *N* from *position*.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/04-Polygon-geometry`

    Hint:
        Avoid costly memory allocations and type conversions by specifying
        primitive properties in the appropriate array type.
    """

    def __init__(self,
                 scene,
                 vertices,
                 position=(0, 0, 0),
                 angle=0,
                 color=(0, 0, 0),
                 rounding_radius=0,
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(position)

        self._geometry = scene.device.module.GeometryPolygon(
            scene._scene, vertices, rounding_radius, N)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width

        self.position[:] = position
        self.angle[:] = angle
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    @property
    def position(self):
        """(N, 2) `Array`: The position of each polygon."""
        return util.Array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def angle(self):
        """(N, ) `Array`: The rotation angle of each polygon (in radians)."""
        return util.Array(self._geometry.getAngleBuffer(), geom=self)

    @property
    def color(self):
        """(N, 2, 3) `Array`: The color of each polygon."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        pos = self.position[:]
        r = self._geometry.getRadius()
        res2d = numpy.array(
            [numpy.min(pos - r, axis=0),
             numpy.max(pos + r, axis=0)])
        res = numpy.array([[res2d[0][0], res2d[0][1], -1e-5],
                           [res2d[1][0], res2d[1][1], 1e-5]])

        return res


class Sphere(Geometry):
    """Sphere geometry.

    Define a set of sphere primitives with individual positions, radii, and
    colors.

    Args:
        scene (Scene): Add the geometry to this scene.

        position ((N, 3) `numpy.ndarray` of ``float32``):
            Position of each sphere.

        radius ((N, ) `numpy.ndarray` of ``float32``):
            Radius of each sphere.

        color ((N, 3) `numpy.ndarray` of ``float32``): Color of each sphere.

        N (int): Number of spheres in the geometry. If ``None``, determine *N*
            from *position*.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/00-Sphere-geometry`

    Hint:
        Avoid costly memory allocations and type conversions by specifying
        primitive properties in the appropriate array type.

    Tip:
        When all spheres are the same size, pass a single value for *radius* and
        numpy will broadcast it to all elements of the array.
    """

    def __init__(self,
                 scene,
                 position=(0, 0, 0),
                 radius=0.5,
                 color=(0, 0, 0),
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(position)

        self._geometry = scene.device.module.GeometrySphere(scene._scene, N)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width

        self.position[:] = position
        self.radius[:] = radius
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        pos = self.position[:]
        r = self.radius[:]
        r = r.reshape(len(r), 1)
        res = numpy.array(
            [numpy.min(pos - r, axis=0),
             numpy.max(pos + r, axis=0)])
        return res

    @property
    def position(self):
        """(N, 3) `Array`: The position of each sphere."""
        return util.Array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def radius(self):
        """(N, ) `Array`: The radius of each sphere."""
        return util.Array(self._geometry.getRadiusBuffer(), geom=self)

    @property
    def color(self):
        """(N, 3) `Array`: The color of each sphere."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)


class Ellipsoid(Geometry):
    """Ellipsoid geometry,





    """
    def __init__(self,
                 scene,
                 position=(0, 0, 0),
                 radius=0.5,
                 color=(0, 0, 0),
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(position)

        self._geometry = scene.device.module.GeometrySphere(scene._scene, N)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width

        self.position[:] = position
        self.radius[:] = radius
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        pos = self.position[:]
        r = self.radius[:]
        r = r.reshape(len(r), 1)
        res = numpy.array(
            [numpy.min(pos - r, axis=0),
             numpy.max(pos + r, axis=0)])
        return res

    @property
    def position(self):
        """(N, 3) `Array`: The position of each sphere."""
        return util.Array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def radius(self):
        """(N, ) `Array`: The radius of each sphere."""
        return util.Array(self._geometry.getRadiusBuffer(), geom=self)

    @property
    def color(self):
        """(N, 3) `Array`: The color of each sphere."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)

    
# end


class Mesh(Geometry):
    """Mesh geometry.

    Define a set of triangle mesh primitives with individual positions,
    orientations, and colors.

    Args:
        scene (Scene): Add the geometry to this scene.

        vertices ((3T, 3) `numpy.ndarray` of ``float32``):
            Vertices of the triangles, listed contiguously. Vertices 0,1,2
            define the first triangle, 3,4,5 define the second, and so on.

        color ((3T, 3) `numpy.ndarray` of ``float32``):
            Color of each vertex.

        position ((N, 3) `numpy.ndarray` of ``float32``):
            Position of each mesh instance.

        orientation ((N, 4) `numpy.ndarray` of ``float32``):
            Orientation of each mesh instance (as a quaternion).

        N (int): Number of mesh instances in the geometry. If ``None``,
            determine *N* from *position*.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/03-Mesh-geometry`

    Hint:
        Avoid costly memory allocations and type conversions by specifying
        primitive properties in the appropriate array type.
    """

    def __init__(self,
                 scene,
                 vertices,
                 position=(0, 0, 0),
                 orientation=(1, 0, 0, 0),
                 color=(0, 0, 0),
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(position)

        self.vertices = numpy.asarray(vertices, dtype=numpy.float32)
        self._geometry = scene.device.module.GeometryMesh(
            scene._scene, self.vertices, N)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width

        self.position[:] = position
        self.orientation[:] = orientation
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    @property
    def position(self):
        """(N, 3) `Array`: The position of each mesh."""
        return util.Array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def orientation(self):
        """(N, 4) `Array`: The orientation of each mesh."""
        return util.Array(self._geometry.getOrientationBuffer(), geom=self)

    @property
    def color(self):
        """(N, 3) `Array`: The color of each sphere."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        a = self.vertices[:, 0]
        b = self.vertices[:, 1]
        c = self.vertices[:, 2]
        r = numpy.array([
            numpy.min([
                numpy.min(a, axis=0),
                numpy.min(b, axis=0),
                numpy.min(c, axis=0)
            ],
                      axis=0),
            numpy.max([
                numpy.max(a, axis=0),
                numpy.max(b, axis=0),
                numpy.max(c, axis=0)
            ],
                      axis=0)
        ])

        pos = self.position[:]
        res = numpy.array(
            [numpy.min(pos + r[0], axis=0),
             numpy.max(pos + r[1], axis=0)])
        return res


class ConvexPolyhedron(Geometry):
    """Convex polyhedron geometry.

    Define a set of convex polyhedron primitives with individual positions,
    orientations, and colors.

    A convex polyhedron is defined by *P* outward facing planes (origin and
    normal vector) and a radius that encompass the shape. Use
    `convex_polyhedron_from_vertices` to construct this from the convex hull of
    a set of vertices.

    Args:
        scene (Scene): Add the geometry to this scene.

        polyhedron_info (Dict): A dictionary containing the face normals
            (``face_normal``), origins (``face_origin``), face colors
            (``face_color``), and the radius (``radius``)).

        position ((N, 3) `numpy.ndarray` of ``float32``):
            Position of each polyhedron instance.

        orientation ((N, 4) `numpy.ndarray` of ``float32``):
            Orientation of each polyhedron instance (as a quaternion).

        color ((N, 3) `numpy.ndarray` of ``float32``):
            Color of each polyhedron.

        N (int): Number of spheres in the geometry. If ``None``, determine *N*
            from *position*.

    See Also:
        Tutorials:

        - :doc:`examples/01-Primitives/02-Convex-polyhedron-geometry`

    Hint:
        Avoid costly memory allocations and type conversions by specifying
        primitive properties in the appropriate array type.
    """

    def __init__(self,
                 scene,
                 polyhedron_info,
                 position=(0, 0, 0),
                 orientation=(1, 0, 0, 0),
                 color=(0, 0, 0),
                 N=None,
                 material=material.Material(solid=1.0, color=(1, 0, 1)),
                 outline_material=material.Material(solid=1.0, color=(0, 0, 0)),
                 outline_width=0.0):
        if N is None:
            N = len(position)

        origins = polyhedron_info['face_origin']
        normals = polyhedron_info['face_normal']
        face_colors = polyhedron_info['face_color']
        r = polyhedron_info['radius']
        self._geometry = scene.device.module.GeometryConvexPolyhedron(
            scene._scene, origins, normals, face_colors, N, r)
        self.material = material
        self.outline_material = outline_material
        self.outline_width = outline_width
        self._radius = r

        self.position[:] = position
        self.orientation[:] = orientation
        self.color[:] = color

        self.scene = scene
        self.scene.geometry.append(self)

    def get_extents(self):
        """Get the extents of the geometry.

        Returns:
            (3,2) `numpy.ndarray` of ``float32``: The lower left and\
                upper right corners of the scene.
        """
        pos = self.position[:]
        r = self._radius
        res = numpy.array(
            [numpy.min(pos - r, axis=0),
             numpy.max(pos + r, axis=0)])
        return res

    @property
    def position(self):
        """(N, 3) `Array`: The position of each polyhedron."""
        return util.Array(self._geometry.getPositionBuffer(), geom=self)

    @property
    def orientation(self):
        """(N, 4) `Array`: The orientation of each polyhedron."""
        return util.Array(self._geometry.getOrientationBuffer(), geom=self)

    @property
    def color(self):
        """(N, 3) `Array`: The color of each polyhedron."""
        return util.Array(self._geometry.getColorBuffer(), geom=self)

    @property
    def color_by_face(self):
        """float: Mix face colors with the per-polyhedron color.

        Set to 0 to color particles by the per-particle `color`. Set to 1 to
        color faces by the per-face color. Set to a value between 0 and 1 to
        blend between the two colors.
        """
        return self._geometry.getColorByFace()

    @color_by_face.setter
    def color_by_face(self, v):
        self._geometry.setColorByFace(v)
