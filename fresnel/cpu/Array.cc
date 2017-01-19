// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Array.h"

namespace fresnel { namespace cpu {

void export_Array(pybind11::module& m)
    {
    pybind11::class_<Array< RGBA<float> >, std::shared_ptr<Array< RGBA<float> >> >(m, "ArrayRGBAf")
        .def_buffer([](Array< RGBA<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< RGBA<float> >::map_py)
        .def("unmap", &Array< RGBA<float> >::unmap)
        ;

    pybind11::class_<Array< RGBA<unsigned char> >, std::shared_ptr<Array< RGBA<unsigned char> >> >(m, "ArrayRGBAc")
        .def_buffer([](Array< RGBA<unsigned char> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< RGBA<unsigned char> >::map_py)
        .def("unmap", &Array< RGBA<unsigned char> >::unmap)
        ;

    pybind11::class_<Array< RGB<float> >, std::shared_ptr<Array< RGB<float> >> >(m, "ArrayRGBf")
        .def_buffer([](Array< RGB<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< RGB<float> >::map_py)
        .def("unmap", &Array< RGB<float> >::unmap)
        ;

    pybind11::class_<Array< RGB<unsigned char> >, std::shared_ptr<Array< RGB<unsigned char> >> >(m, "ArrayRGBc")
        .def_buffer([](Array< RGB<unsigned char> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< RGB<unsigned char> >::map_py)
        .def("unmap", &Array< RGB<unsigned char> >::unmap)
        ;

    pybind11::class_<Array< vec3<float> >, std::shared_ptr<Array< vec3<float> >> >(m, "ArrayVec3f")
        .def_buffer([](Array< vec3<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< vec3<float> >::map_py)
        .def("unmap", &Array< vec3<float> >::unmap)
        ;

    pybind11::class_<Array< vec2<float> >, std::shared_ptr<Array< vec2<float> >> >(m, "ArrayVec2f")
        .def_buffer([](Array< vec2<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< vec2<float> >::map_py)
        .def("unmap", &Array< vec2<float> >::unmap)
        ;

    pybind11::class_<Array< quat<float> >, std::shared_ptr<Array< quat<float> >> >(m, "ArrayQuat3f")
        .def_buffer([](Array< quat<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< quat<float> >::map_py)
        .def("unmap", &Array< quat<float> >::unmap)
        ;

    pybind11::class_<Array< float >, std::shared_ptr<Array< float >> >(m, "Array_f")
        .def_buffer([](Array< float > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        .def("map", &Array< float >::map_py)
        .def("unmap", &Array< float >::unmap)
        ;
    }

} }
