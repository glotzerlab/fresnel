// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <optixu/optixu_math_namespace.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometryCylinder.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of spheres in the geometry
*/
GeometryCylinder::GeometryCylinder(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
    {
    // Declared initial variables
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();
    m_geometry->setPrimitiveCount(N);

    const char* path_to_ptx = "GeometryCylinder.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    optix::Buffer optix_points
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 2, N);
    optix::Buffer optix_radius = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
    optix::Buffer optix_color
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 2, N);

    m_geometry["cylinder_points"]->setBuffer(optix_points);
    m_geometry["cylinder_radius"]->setBuffer(optix_radius);
    m_geometry["cylinder_color"]->setBuffer(optix_color);

    // intialize python access to buffers
    m_points = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(2, optix_points));
    m_radius = std::shared_ptr<Array<float>>(new Array<float>(1, optix_radius));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(2, optix_color));
    setupInstance();
    }

GeometryCylinder::~GeometryCylinder() { }

void export_GeometryCylinder(pybind11::module& m)
    {
    pybind11::class_<GeometryCylinder, Geometry, std::shared_ptr<GeometryCylinder>>(
        m,
        "GeometryCylinder")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPointsBuffer", &GeometryCylinder::getPointsBuffer)
        .def("getRadiusBuffer", &GeometryCylinder::getRadiusBuffer)
        .def("getColorBuffer", &GeometryCylinder::getColorBuffer);
    }

    } // namespace gpu
    } // namespace fresnel
