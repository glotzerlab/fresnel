// Copyright (c) 2016-2022 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "common/Camera.h"
#include "common/ColorMath.h"
#include "common/ConvexPolyhedronBuilder.h"
#include "common/Light.h"
#include "common/Material.h"
#include "common/VectorMath.h"

#include <sstream>

using namespace std;
using namespace fresnel;

bool gpu_built()
    {
#ifdef FRESNEL_BUILD_GPU
    return true;
#else
    return false;
#endif
    }

bool cpu_built()
    {
#ifdef FRESNEL_BUILD_CPU
    return true;
#else
    return false;
#endif
    }

PYBIND11_MODULE(_common, m)
    {
    m.def("gpu_built", &gpu_built);
    m.def("cpu_built", &cpu_built);
    m.def("find_polyhedron_faces", &find_polyhedron_faces);

    pybind11::class_<RGB<float>>(m, "RGBf")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("r", &RGB<float>::r)
        .def_readwrite("g", &RGB<float>::g)
        .def_readwrite("b", &RGB<float>::b);

    pybind11::class_<Material>(m, "Material")
        .def(pybind11::init<>())
        .def_readwrite("solid", &Material::solid)
        .def_readwrite("primitive_color_mix", &Material::primitive_color_mix)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("specular", &Material::specular)
        .def_readwrite("spec_trans", &Material::spec_trans)
        .def_readwrite("metal", &Material::metal)
        .def_readwrite("color", &Material::color)
        .def("__repr__",
             [](const Material& a)
             {
                 ostringstream s;
                 s << "<fresnel._common.Material:"
                   << " solid=" << a.solid << " color=(" << a.color.r << ", " << a.color.g << ", "
                   << a.color.b << ")"
                   << " primitive_color_mix=" << a.primitive_color_mix
                   << " roughness=" << a.roughness << " specular=" << a.specular
                   << " spec_trans=" << a.spec_trans << " metal=" << a.metal << ">";

                 return s.str();
             });

    pybind11::class_<vec3<float>>(m, "vec3f")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("x", &vec3<float>::x)
        .def_readwrite("y", &vec3<float>::y)
        .def_readwrite("z", &vec3<float>::z)
        .def("__repr__",
             [](const vec3<float>& a)
             {
                 ostringstream s;
                 s << "<fresnel._common.vec3f (" << a.x << ", " << a.y << ", " << a.z << ")>";
                 return s.str();
             });

    pybind11::class_<UserCamera>(m, "UserCamera")
        .def(pybind11::init<>())
        .def_readwrite("position", &UserCamera::position)
        .def_readwrite("look_at", &UserCamera::look_at)
        .def_readwrite("up", &UserCamera::up)
        .def_readwrite("h", &UserCamera::h)
        .def_readwrite("f", &UserCamera::f)
        .def_readwrite("f_stop", &UserCamera::f_stop)
        .def_readwrite("focus_distance", &UserCamera::focus_distance)
        .def_readwrite("model", &UserCamera::model);

    pybind11::enum_<CameraModel>(m, "CameraModel")
        .value("orthographic", CameraModel::orthographic)
        .value("perspective", CameraModel::perspective);

    pybind11::class_<CameraBasis>(m, "CameraBasis")
        .def(pybind11::init<const UserCamera&>())
        .def_readwrite("u", &CameraBasis::u)
        .def_readwrite("v", &CameraBasis::v)
        .def_readwrite("w", &CameraBasis::w);

    pybind11::class_<Lights>(m, "Lights")
        .def(pybind11::init<>())
        .def_readwrite("N", &Lights::N)
        .def("getDirection", &Lights::getDirection)
        .def("setDirection", &Lights::setDirection)
        .def("getColor", &Lights::getColor)
        .def("setColor", &Lights::setColor)
        .def("getTheta", &Lights::getTheta)
        .def("setTheta", &Lights::setTheta);
    }
