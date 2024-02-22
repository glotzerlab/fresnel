// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOMETRY_MESH_H_
#define GEOMETRY_MESH_H_

#include "embree_platform.h"
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace cpu
    {
//! Mesh geometry
/*! Define a triangulated mesh geometry.

    A triangulated mesh is defined as a list of vertices, and an array of indices pointing to the
   vertices for each triangle.

    The triangles must be oriented with an outward facing normal, i.e. in triangle indices in
   counter-clockwise direction.
*/
class GeometryMesh : public Geometry
    {
    public:
    //! Constructor
    GeometryMesh(
        std::shared_ptr<Scene> scene,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
        unsigned int N);
    //! Destructor
    virtual ~GeometryMesh();

    //! Get the position buffer
    std::shared_ptr<Array<vec3<float>>> getPositionBuffer()
        {
        return m_position;
        }

    //! Get the orientation buffer
    std::shared_ptr<Array<quat<float>>> getOrientationBuffer()
        {
        return m_orientation;
        }

    //! Get the color buffer
    std::shared_ptr<Array<RGB<float>>> getColorBuffer()
        {
        return m_color;
        }

    protected:
    std::vector<vec3<float>>
        m_vertices; //!< Holds the vertex coordinates in ccw order for each face
    std::shared_ptr<Array<RGB<float>>> m_color; //!< Color for each vertex point

    std::shared_ptr<Array<vec3<float>>> m_position; //!< Position of each polyhedron
    std::shared_ptr<Array<quat<float>>> m_orientation; //!< Orientation of each polyhedron

    unsigned int m_N; //!< Number of polyhedra

    //! Embree bounding function
    static void bounds(const struct RTCBoundsFunctionArguments* args);

    //! Embree ray intersection function
    static void intersect(const struct RTCIntersectFunctionNArguments* args);
    };

//! Export GeometryMesh to python
void export_GeometryMesh(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
