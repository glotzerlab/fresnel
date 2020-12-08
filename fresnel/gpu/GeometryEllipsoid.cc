// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optixu/optixu_math_namespace.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometryEllipsoid.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of ellipsoids in the geometry
*/
GeometryEllipsoid::GeometryEllipsoid(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
    {
    // Declared initial variables
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();
    m_geometry->setPrimitiveCount(N);

    const char* path_to_ptx = "GeometryEllipsoid.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    optix::Buffer optix_positions
        = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
    optix::Buffer optix_radii = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

    m_geometry["ellipsoid_position"]->setBuffer(optix_positions);
    m_geometry["ellipsoid_radii"]->setBuffer(optix_radii);
	m_geometry["ellipsoid_orientation"]->setBuffer(optix_orientation);
    m_geometry["ellipsoid_color"]->setBuffer(optix_color);

    // intialize python access to buffers
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(1, optix_positions));
    m_radii = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(1, optix_radii));
    m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>>(1, optix_orientation));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(1, optix_color));
    setupInstance();
    }

GeometryEllipsoid::~GeometryEllipsoid() { }

void export_GeometryEllipsoid(pybind11::module& m)
    {
    pybind11::class_<GeometryEllipsoid, Geometry, std::shared_ptr<GeometryEllipsoid>>(m, "GeometryEllipsoid")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometryEllipsoid::getPositionBuffer)
        .def("getRadiiBuffer", &GeometryEllipsoid::getRadiiBuffer)
	    .def("getOrientationBuffer", &GeometryEllipsoid::getOrientationBuffer)
        .def("getColorBuffer", &GeometryEllipsoid::getColorBuffer);
    }

    } // namespace gpu
    } // namespace fresnel
