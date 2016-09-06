// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "Device.h"
#include "Scene.h"
// #include "Geometry.h"
// #include "GeometryTriangleMesh.h"
// #include "Tracer.h"
// #include "TracerWhitted.h"

#include <sstream>

using namespace fresnel::gpu;
using namespace std;

PYBIND11_PLUGIN(_gpu)
    {
    pybind11::module m("_gpu");

    export_Device(m);
    export_Scene(m);
    // export_Geometry(m);
    // export_GeometryTriangleMesh(m);
    // export_Tracer(m);
    // export_TracerWhitted(m);

    return m.ptr();
    }
