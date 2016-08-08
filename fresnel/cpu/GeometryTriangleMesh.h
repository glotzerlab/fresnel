// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_TRIANGLE_MESH_H_
#define GEOMETRY_TRIANGLE_MESH_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Geometry.h"

namespace fresnel { namespace cpu {

//! Triangle mesh geometry
/*! Define a triangle mesh geometry with a list of vertices and a list of indices that index into the
    vertices.

    The initial implementation takes in std::vector's of tuples which will be translated from python
    lists of tuples. This API will be replaced by direct data buffers at some point.
*/
class GeometryTriangleMesh : public Geometry
    {
    public:
        //! Constructor
        GeometryTriangleMesh(std::shared_ptr<Scene> scene,
                             const std::vector<std::tuple<float, float, float> > &vertices,
                             const std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > &indices);
        //! Destructor
        virtual ~GeometryTriangleMesh();

    };

//! Export GeometryTriangleMesh to python
void export_GeometryTriangleMesh(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
