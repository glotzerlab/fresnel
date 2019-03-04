// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "Device.h"
#include "Scene.h"
#include "Geometry.h"
#include "GeometryCylinder.h"
#include "GeometryMesh.h"
#include "GeometryPolygon.h"
#include "GeometryConvexPolyhedron.h"
#include "GeometrySphere.h"
#include "Tracer.h"
#include "TracerDirect.h"
#include "TracerPath.h"

#include <sstream>

using namespace fresnel::gpu;
using namespace std;

unsigned int get_num_available_devices()
    {
    return optix::Context::getDeviceCount();
    }

PYBIND11_MODULE(_gpu, m)
    {
    m.def("get_num_available_devices", &get_num_available_devices);

    export_Device(m);
    export_Scene(m);
    export_Geometry(m);
    export_GeometryCylinder(m);
    export_GeometryMesh(m);
    export_GeometryPolygon(m);
    export_GeometryConvexPolyhedron(m);
    export_GeometrySphere(m);
    export_Tracer(m);
    export_TracerDirect(m);
    export_TracerPath(m);
    export_Array(m);
    }
