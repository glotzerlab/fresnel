// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>
#include <optixu/optixu_math_namespace.h>

#include "GeometryHemisphere.h"

namespace fresnel { namespace gpu {

    /*! \param scene Scene to attach the Geometry to
        \param N number of spheres in the geometry
    */
    GeometryHemisphere::GeometryHemisphere(std::shared_ptr<Scene> scene, unsigned int N)
        : Geometry(scene)
        {

        // Declared initial variabls
        optix::Program intersection_program;
        optix::Program bounding_box_program;

        auto device = scene->getDevice();
        auto context = device->getContext();
        m_geometry = context->createGeometry();
        m_geometry->setPrimitiveCount(N);

        const char * path_to_ptx = "GeometryHemisphere.ptx";
        bounding_box_program = device->getProgram(path_to_ptx, "bounds");
        m_geometry->setBoundingBoxProgram(bounding_box_program);

        intersection_program = device->getProgram(path_to_ptx, "intersect");
        m_geometry->setIntersectionProgram(intersection_program);

        optix::Buffer optix_positions = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
        optix::Buffer optix_orientations = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, N);
        optix::Buffer optix_radius = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
        optix::Buffer optix_directors = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);
        optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

        m_geometry["hemisphere_position"]->setBuffer(optix_positions);
        m_geometry["hemisphere_orientation"]->setBuffer(optix_orientations);
        m_geometry["hemisphere_radius"]->setBuffer(optix_radius);
        m_geometry["hemisphere_director"]->setBuffer(optix_directors);
        m_geometry["hemisphere_color"]->setBuffer(optix_color);

        // intialize python access to buffers
        m_position = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(1, optix_positions));
        m_orientation = std::shared_ptr< Array< quat<float> > >(new Array< quat<float> >(1, optix_orientations));
        m_radius = std::shared_ptr< Array< float > >(new Array< float >(1, optix_radius));
        m_director = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(1, optix_directors));
        m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(1, optix_color));
        setupInstance();
        }

    GeometryHemisphere::~GeometryHemisphere()
        {
        }

    void export_GeometryHemisphere(pybind11::module& m)
        {
        pybind11::class_<GeometryHemisphere, Geometry, std::shared_ptr<GeometryHemisphere>>(m, "GeometryHemisphere")
            .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
            .def("getPositionBuffer", &GeometryHemisphere::getPositionBuffer)
            .def("getOrientationBuffer", &GeometryHemisphere::getOrientationBuffer)
            .def("getRadiusBuffer", &GeometryHemisphere::getRadiusBuffer)
            .def("getDirectorBuffer", &GeometryHemisphere::getDirectorBuffer)
            .def("getColorBuffer", &GeometryHemisphere::getColorBuffer)
            ;
        }

} } // end namespace fresnel::gpu
