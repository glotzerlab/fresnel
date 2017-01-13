// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "Device.h"
#include "Scene.h"
#include "Geometry.h"
#include "GeometryPrism.h"
#include "GeometrySphere.h"
#include "Tracer.h"
#include "TracerWhitted.h"

#include <sstream>

using namespace fresnel::gpu;
using namespace std;

unsigned int get_num_available_devices()
    {
    return optix::Context::getDeviceCount();
    }

PYBIND11_PLUGIN(_gpu)
    {
    pybind11::module m("_gpu");

    m.def("get_num_available_devices", &get_num_available_devices);

    export_Device(m);
    export_Scene(m);
    export_Geometry(m);
    export_GeometryPrism(m);
    export_GeometrySphere(m);
    export_Tracer(m);
    export_TracerWhitted(m);
    export_Array(m);

    return m.ptr();
    }
