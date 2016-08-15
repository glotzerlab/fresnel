// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "Device.h"
#include "Scene.h"
#include "Geometry.h"
#include "GeometryTriangleMesh.h"
#include "Tracer.h"
#include "TracerWhitted.h"
#include "common/Material.h"

using namespace fresnel::cpu;

PYBIND11_PLUGIN(_cpu)
    {
    pybind11::module m("_cpu");

    export_Device(m);
    export_Scene(m);
    export_Geometry(m);
    export_GeometryTriangleMesh(m);
    export_Tracer(m);
    export_TracerWhitted(m);

    pybind11::class_< colorRGB<float> >(m, "ColorRGBfloat")
        .def_readwrite("r", &colorRGB<float>::r)
        .def_readwrite("g", &colorRGB<float>::g)
        .def_readwrite("b", &colorRGB<float>::b);

    pybind11::class_< Material >(m, "Material")
        .def_readwrite("solid", &Material::solid)
        .def_readwrite("color", &Material::color);

    pybind11::class_< vec3<float> >(m, "vec3f")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("x", &vec3<float>::x)
        .def_readwrite("y", &vec3<float>::y)
        .def_readwrite("z", &vec3<float>::z);

    pybind11::class_< Camera >(m, "Camera")
        .def(pybind11::init<vec3<float>, vec3<float>, vec3<float>, vec3<float> >())
        .def_readwrite("p", &Camera::p)
        .def_readwrite("d", &Camera::d)
        .def_readwrite("u", &Camera::u)
        .def_readwrite("u", &Camera::r);

    return m.ptr();
    }
