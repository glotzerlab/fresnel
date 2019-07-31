// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryPolygon.h"

namespace fresnel { namespace gpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the polygon (in counterclockwise order)
    \param position position of each polygon
    \param orientation orientation angle of each polygon
    \param height height of each polygon

    Initialize the polygon.
*/
GeometryPolygon::GeometryPolygon(std::shared_ptr<Scene> scene,
                             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                             float rounding_radius,
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
    m_geometry->setPrimitiveCount(N);

    // load bounding and intresection programs
    const char * path_to_ptx = "GeometryPolygon.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    // copy the vertices from the numpy array to internal storage
    pybind11::buffer_info info = vertices.request();

    if (info.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info.shape[1] != 2)
        throw std::runtime_error("vertices must be a Nvert by 2 array");

    float *verts_f = (float *)info.ptr;

    // set up OptiX data buffers
    m_vertices = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT2, info.shape[0]);

    vec2<float>* optix_vertices = (vec2<float>*)m_vertices->map();

    for (unsigned int i = 0; i < info.shape[0]; i++)
        {
        vec2<float> p0(verts_f[i*2], verts_f[i*2+1]);

        optix_vertices[i] = p0;

        // precompute radius in the xy plane
        m_radius = std::max(m_radius, sqrtf(dot(p0,p0)));
        }

    m_vertices->unmap();

    // pad the radius with the rounding radius
    m_radius += rounding_radius;

    // copy data values to OptiX
    m_geometry["polygon_radius"]->setFloat(m_radius);
    m_geometry["polygon_rounding_radius"]->setFloat(rounding_radius);
    m_geometry["polygon_vertices"]->setBuffer(m_vertices);

    optix::Buffer optix_position = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT2, N);
    optix::Buffer optix_angle = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

    m_geometry["polygon_position"]->setBuffer(optix_position);
    m_geometry["polygon_angle"]->setBuffer(optix_angle);
    m_geometry["polygon_color"]->setBuffer(optix_color);

    // intialize python access to buffers
    m_position = std::make_shared< Array< vec2<float> > >(1, optix_position);
    m_angle = std::make_shared< Array< float > >(1, optix_angle);
    m_color = std::make_shared< Array< RGB<float> > >(1, optix_color);
    setupInstance();

    m_valid = true;
    }

GeometryPolygon::~GeometryPolygon()
    {
    m_vertices->destroy();
    }

/*! \param m Python module to export in
 */
void export_GeometryPolygon(pybind11::module& m)
    {
    pybind11::class_<GeometryPolygon, Geometry, std::shared_ptr<GeometryPolygon> >(m, "GeometryPolygon")
        .def(pybind11::init<std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             float,
             unsigned int>())
        .def("getPositionBuffer", &GeometryPolygon::getPositionBuffer)
        .def("getAngleBuffer", &GeometryPolygon::getAngleBuffer)
        .def("getColorBuffer", &GeometryPolygon::getColorBuffer)
        .def("getRadius", &GeometryPolygon::getRadius)
        ;
    }

} } // end namespace fresnel::gpu
