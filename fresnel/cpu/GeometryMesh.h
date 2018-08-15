// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_MESH_H_
#define GEOMETRY_MESH_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace cpu {

//! Mesh geometry
/*! Define a triangulated mesh geometry.

    A triangulated mesh is defined as a list of vertices, and an array of indices pointing to the vertices for each triangle.

    The triangles must be oriented with an outward facing normal, i.e. in triangle indices in counter-clockwise direction.
*/
class GeometryMesh : public Geometry
    {
    public:
        //! Constructor
        GeometryMesh(std::shared_ptr<Scene> scene, unsigned int N);
        //! Destructor
        virtual ~GeometryMesh();

        //! Get the triangle buffer
        std::shared_ptr< Array< vec3<float> > > getPointsBuffer()
            {
            return m_points;
            }

        //! Get the color buffer
        std::shared_ptr< Array< RGB<float> > > getColorBuffer()
            {
            return m_color;
            }

    protected:
        unsigned int m_N;                           //!< Number of polyhedra

        std::shared_ptr< Array< vec3<float> > > m_points;    //!< Position of the vertices of each triangle
        std::shared_ptr< Array< RGB<float> > > m_color;      //!< Color for each vertex point

        //! Embree bounding function
        static void bounds(const struct RTCBoundsFunctionArguments *args);

        //! Embree ray intersection function
        static void intersect(const struct RTCIntersectFunctionNArguments *args);

    };

//! Export GeometryMesh to python
void export_GeometryMesh(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
