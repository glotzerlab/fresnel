// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>

#include "GeometryTriangleMesh.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices Vertices of the triangles
    \param indicies Indices into the vertices defining each triangle

    Creates a triangle mesh geometry with the given vertices and indices.
*/
GeometryTriangleMesh::GeometryTriangleMesh(std::shared_ptr<Scene> scene,
                                          const std::vector<std::tuple<float, float, float> > &vertices,
                                          const std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > &indices)
    : Geometry(scene)
    {
    // create the geometry
    m_geom_id = rtcNewTriangleMesh(m_scene->getRTCScene(),
                                   RTC_GEOMETRY_DYNAMIC,
                                   indices.size(),
                                   vertices.size(),
                                   1);
    m_device->checkError();

    struct Vertex   { float x, y, z, a; };
    struct Triangle { int v0, v1, v2; };

    // copy the vertex and index information into the Embree buffers
    Vertex* vertices_raw = (Vertex*) rtcMapBuffer(m_scene->getRTCScene(), m_geom_id, RTC_VERTEX_BUFFER);
    for (unsigned int i = 0; i < vertices.size(); i++)
        {
        vertices_raw[i].x = std::get<0>(vertices[i]);
        vertices_raw[i].y = std::get<1>(vertices[i]);
        vertices_raw[i].z = std::get<2>(vertices[i]);
        vertices_raw[i].a = 0;
        }
    rtcUnmapBuffer(m_scene->getRTCScene(), m_geom_id, RTC_VERTEX_BUFFER);
    m_device->checkError();

    Triangle* indices_raw = (Triangle*) rtcMapBuffer(m_scene->getRTCScene(), m_geom_id, RTC_INDEX_BUFFER);
    for (unsigned int i = 0; i < indices.size(); i++)
        {
        indices_raw[i].v0 = std::get<0>(indices[i]);
        indices_raw[i].v1 = std::get<1>(indices[i]);
        indices_raw[i].v2 = std::get<2>(indices[i]);
        }
    rtcUnmapBuffer(m_scene->getRTCScene(), m_geom_id, RTC_INDEX_BUFFER);
    m_device->checkError();
    }

GeometryTriangleMesh::~GeometryTriangleMesh()
    {
    }

/*! \param m Python module to export in
 */
void export_GeometryTriangleMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryTriangleMesh, std::shared_ptr<GeometryTriangleMesh> >(m, "GeometryTriangleMesh", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>, const std::vector<std::tuple<float, float, float> > &, const std::vector<std::tuple<unsigned int, unsigned int, unsigned int> > & >())
        ;
    }

} } // end namespace fresnel::cpu
