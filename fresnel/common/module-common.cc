// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "common/Material.h"
#include "common/Camera.h"
#include "common/VectorMath.h"
#include "common/ColorMath.h"

#include <sstream>

// TODO: Put common code into fresnel::common workspace?
using namespace std;

PYBIND11_PLUGIN(_common)
    {
    pybind11::module m("_common");

    pybind11::class_< RGB<float> >(m, "RGBf")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("r", &RGB<float>::r)
        .def_readwrite("g", &RGB<float>::g)
        .def_readwrite("b", &RGB<float>::b);

    pybind11::class_< Material, std::shared_ptr<Material> >(m, "Material")
        .def(pybind11::init<>())
        .def_readwrite("solid", &Material::solid)
        .def_readwrite("color", &Material::color)
        .def("__repr__",
            [](const Material &a)
                {
                ostringstream s;
                s << "<fresnel._common.Material:"
                  << " solid=" << a.solid
                  << " color=(" << a.color.r << ", " << a.color.g << ", " << a.color.b << ")"
                  << ">";

                return s.str();
                }
            )
        ;

    pybind11::class_< vec3<float> >(m, "vec3f")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("x", &vec3<float>::x)
        .def_readwrite("y", &vec3<float>::y)
        .def_readwrite("z", &vec3<float>::z)
        .def("__repr__",
            [](const vec3<float> &a)
                {
                ostringstream s;
                s << "<fresnel._common.vec3f (" << a.x << ", " << a.y << ", " << a.z << ")>";
                return s.str();
                }
            )
        ;

    pybind11::class_< Camera >(m, "Camera")
        .def(pybind11::init<const vec3<float>&, const vec3<float>&, const vec3<float>&, float >())
        .def_readwrite("p", &Camera::p)
        .def_readwrite("d", &Camera::d)
        .def_readwrite("u", &Camera::u)
        .def_readwrite("r", &Camera::r)
        .def_readwrite("h", &Camera::h)
        ;

    return m.ptr();
    }
