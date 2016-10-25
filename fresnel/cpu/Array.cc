// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Array.h"

namespace fresnel { namespace cpu {

void export_Array(pybind11::module& m)
    {
    pybind11::class_<Array< RGBA<float> >, std::shared_ptr<Array< RGBA<float> >> >(m, "ArrayRGBf")
        .def_buffer([](Array< RGBA<float> > &t) -> pybind11::buffer_info { return t.getBuffer(); })
        ;
    }

} }
