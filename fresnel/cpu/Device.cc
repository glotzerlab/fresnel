// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <memory>

#include "Device.h"

namespace fresnel { namespace cpu {

Device::Device()
    {
    m_device = rtcNewDevice(nullptr);

    if (m_device == nullptr)
        {
        throw std::runtime_error("Error creating embree device");
        }
    }

Device::~Device()
    {
    rtcDeleteDevice(m_device);
    }

void export_Device(pybind11::module& m)
    {
    pybind11::class_<Device, std::shared_ptr<Device> >(m, "Device")
        .def(pybind11::init<>())
        ;
    }

} } // end namespace fresnel::cpu
