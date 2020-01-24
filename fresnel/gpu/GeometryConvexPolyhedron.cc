// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryConvexPolyhedron.h"

namespace fresnel { namespace gpu {

/*! \param scene Scene to attach the Geometry to
    \param plane_origins Origins of the planes that make up the polyhedron
    \param plane_normals Normals of the planes that make up the polyhedron
    \param r radius of the polyhedron

    Initialize the polyhedron geometry.
*/
GeometryConvexPolyhedron::GeometryConvexPolyhedron(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_origins,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_normals,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_colors,
    unsigned int N,
    float r)
    : Geometry(scene)
{
    // create the geometry
    // intersection and bounding programs are not stored for later destruction, as Device will
    // destroy its program cache
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();
    m_geometry->setPrimitiveCount(N);

    // load bounding and intresection programs
    const char* path_to_ptx = "GeometryConvexPolyhedron.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    // access the plane data
    pybind11::buffer_info info_origin = plane_origins.request();

    if (info_origin.ndim != 2)
        throw std::runtime_error("plane_origins must be a 2-dimensional array");

    if (info_origin.shape[1] != 3)
        throw std::runtime_error("plane_origins must be a Nvert by 3 array");

    float* origin_f = (float*)info_origin.ptr;

    pybind11::buffer_info info_normal = plane_normals.request();

    if (info_normal.ndim != 2)
        throw std::runtime_error("plane_normals must be a 2-dimensional array");

    if (info_normal.shape[1] != 3)
        throw std::runtime_error("plane_normals must be a Nvert by 3 array");

    if (info_normal.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and normal arrays");

    float* normal_f = (float*)info_normal.ptr;

    pybind11::buffer_info info_color = plane_colors.request();

    if (info_color.ndim != 2)
        throw std::runtime_error("plane_colors must be a 2-dimensional array");

    if (info_color.shape[1] != 3)
        throw std::runtime_error("plane_colors must be a Nvert by 3 array");

    if (info_color.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and color arrays");

    float* color_f = (float*)info_color.ptr;

    // copy data values to OptiX
    m_geometry["convex_polyhedron_radius"]->setFloat(r);

    // set up OptiX data buffers
    m_plane_origin
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, info_normal.shape[0]);
    m_plane_normal
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, info_normal.shape[0]);
    m_plane_color
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, info_normal.shape[0]);

    vec3<float>* optix_plane_origin = (vec3<float>*)m_plane_origin->map();
    vec3<float>* optix_plane_normal = (vec3<float>*)m_plane_normal->map();
    RGB<float>* optix_plane_color = (RGB<float>*)m_plane_color->map();

    // construct planes in C++ data structures
    for (unsigned int i = 0; i < info_normal.shape[0]; i++)
    {
        vec3<float> n(normal_f[i * 3], normal_f[i * 3 + 1], normal_f[i * 3 + 2]);
        n = n / sqrtf(dot(n, n));

        optix_plane_origin[i]
            = vec3<float>(origin_f[i * 3], origin_f[i * 3 + 1], origin_f[i * 3 + 2]);
        optix_plane_normal[i] = n;
        optix_plane_color[i] = RGB<float>(color_f[i * 3], color_f[i * 3 + 1], color_f[i * 3 + 2]);
    }

    m_plane_origin->unmap();
    m_plane_normal->unmap();
    m_plane_color->unmap();

    m_geometry["convex_polyhedron_plane_origin"]->setBuffer(m_plane_origin);
    m_geometry["convex_polyhedron_plane_normal"]->setBuffer(m_plane_normal);
    m_geometry["convex_polyhedron_plane_color"]->setBuffer(m_plane_color);

    optix::Buffer optix_position
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
    optix::Buffer optix_orientation
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

    m_geometry["convex_polyhedron_position"]->setBuffer(optix_position);
    m_geometry["convex_polyhedron_orientation"]->setBuffer(optix_orientation);
    m_geometry["convex_polyhedron_color"]->setBuffer(optix_color);

    // initialize python access to buffers
    m_position = std::make_shared<Array<vec3<float>>>(1, optix_position);
    m_orientation = std::make_shared<Array<quat<float>>>(1, optix_orientation);
    m_color = std::make_shared<Array<RGB<float>>>(1, optix_color);
    setupInstance();

    m_valid = true;
}

GeometryConvexPolyhedron::~GeometryConvexPolyhedron()
{
    m_plane_origin->destroy();
    m_plane_normal->destroy();
    m_plane_color->destroy();
}

/*! \param m Python module to export in
 */
void export_GeometryConvexPolyhedron(pybind11::module& m)
{
    pybind11::class_<GeometryConvexPolyhedron, Geometry, std::shared_ptr<GeometryConvexPolyhedron>>(
        m,
        "GeometryConvexPolyhedron")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int,
             float>())
        .def("getPositionBuffer", &GeometryConvexPolyhedron::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryConvexPolyhedron::getOrientationBuffer)
        .def("getColorBuffer", &GeometryConvexPolyhedron::getColorBuffer)
        .def("setColorByFace", &GeometryConvexPolyhedron::setColorByFace)
        .def("getColorByFace", &GeometryConvexPolyhedron::getColorByFace);
}

}} // namespace fresnel::gpu
