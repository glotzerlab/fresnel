// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>

#include "Geometry.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices Vertices of the triangles
    \param indicies Indices into the vertices defining each triangle

    Creates a triangle mesh geometry with the given vertices and indices.
*/
GeomtryTriangleMesh::GeometryTriangleMesh(std::shared_ptr<Scene> scene,
                                          const std::vector<std:tuple<float, float, float> > &vertices,
                                          const std::vector<std:tuple<unsigned int, unsigned int, unsigned int> > &indices)
    : Geometry(scene)
    {
    }

GeometryTriangleMesh::~GeometryTriangleMesh()
    {
    }

/*! \param m Python module to export in
 */
void export_GeometryTriangleMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryTriangleMesh, std::shared_ptr<GeometryTriangleMesh> >(m, "GeometryTriangleMesh", py::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene> >())
        ;
    }

} } // end namespace fresnel::cpu
