// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <iostream>

#include "GeometryMesh.h"

namespace fresnel { namespace gpu {

/*! \param scene Scene to attach the Geometry to
    \param plane_origins Origins of the planes that make up the polyhedron
    \param plane_normals Normals of the planes that make up the polyhedron

    Initialize the polyhedron geometry.
*/
GeometryMesh::GeometryMesh(std::shared_ptr<Scene> scene,
                           pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                           unsigned int N)
    : Geometry(scene)
    {
    // create the geometry
    // intersection and bounding programs are not stored for later destruction, as Device will destroy its program cache
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();

    // load bounding and intresection programs
    const char * path_to_ptx = "_ptx_generated_GeometryMesh.cu.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    // access the triangle vertices
    pybind11::buffer_info info_vertices = vertices.request();

    if (info_vertices.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nvert by 3 array");

    if (info_vertices.shape[0] % 3 != 0)
        throw std::runtime_error("the number of triangle vertices must be a multiple of three.");

    unsigned int n_faces = info_vertices.shape[0] / 3;
    unsigned int n_verts = info_vertices.shape[0];
    m_geometry->setPrimitiveCount(N*n_faces);

    float *verts_f = (float *)info_vertices.ptr;

    // set up OptiX data buffers
    m_vertices = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, n_verts);

    vec3<float>* optix_vertices = (vec3<float>*)m_vertices->map();
    memcpy(optix_vertices, verts_f, n_verts*3*sizeof(float));
    m_vertices->unmap();

    m_geometry["mesh_vertices"]->setBuffer(m_vertices);

    optix::Buffer optix_position = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
    optix::Buffer optix_orientation = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, n_verts);

    m_geometry["mesh_position"]->setBuffer(optix_position);
    m_geometry["mesh_orientation"]->setBuffer(optix_orientation);
    m_geometry["mesh_color"]->setBuffer(optix_color);

    // initialize python access to buffers
    m_position = std::make_shared< Array< vec3<float> > >(1, optix_position);
    m_orientation = std::make_shared< Array< quat<float> > >(1, optix_orientation);
    m_color = std::make_shared< Array< RGB<float> > >(1, optix_color);
    setupInstance();
    }

GeometryMesh::~GeometryMesh()
    {
    m_vertices->destroy();
    }

/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, std::shared_ptr<GeometryMesh> >(m, "GeometryMesh", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryMesh::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryMesh::getOrientationBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer)
        ;
    }

} } // end namespace fresnel::cpu
