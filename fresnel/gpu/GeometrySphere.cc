// Copyright (c) 2016-2021 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <optixu/optixu_math_namespace.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometrySphere.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of spheres in the geometry
*/
GeometrySphere::GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
    {
    // Declared initial variabls
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();
    m_geometry->setPrimitiveCount(N);

    const char* path_to_ptx = "GeometrySphere.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    optix::Buffer optix_positions
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
    optix::Buffer optix_radius = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

    m_geometry["sphere_position"]->setBuffer(optix_positions);
    m_geometry["sphere_radius"]->setBuffer(optix_radius);
    m_geometry["sphere_color"]->setBuffer(optix_color);

    // intialize python access to buffers
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(1, optix_positions));
    m_radius = std::shared_ptr<Array<float>>(new Array<float>(1, optix_radius));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(1, optix_color));
    setupInstance();
    }

GeometrySphere::~GeometrySphere() { }

void export_GeometrySphere(pybind11::module& m)
    {
    pybind11::class_<GeometrySphere, Geometry, std::shared_ptr<GeometrySphere>>(m, "GeometrySphere")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometrySphere::getPositionBuffer)
        .def("getRadiusBuffer", &GeometrySphere::getRadiusBuffer)
        .def("getColorBuffer", &GeometrySphere::getColorBuffer);
    }

    } // namespace gpu
    } // namespace fresnel
