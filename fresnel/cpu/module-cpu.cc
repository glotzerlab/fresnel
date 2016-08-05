// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "Device.h"

using namespace fresnel::cpu;

PYBIND11_DECLARE_HOLDER_TYPE(T_shared_ptr_bind, std::shared_ptr<T_shared_ptr_bind>);

PYBIND11_PLUGIN(_cpu)
    {
    pybind11::module m("_cpu");

    export_Device(m);

    return m.ptr();
    }
