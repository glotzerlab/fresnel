// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <memory>
#include <string>

#include "Device.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! Construct a new RTcontext with default options.
*/
Device::Device()
    {
    std::cout << "Create GPU Device" << std::endl;
    m_context = optix::Context::create();
    }

/*! Destroy the underlying RTcontext
*/
Device::~Device()
    {
    std::cout << "Destroy GPU Device" << std::endl;
    m_context->destroy();
    }

/*! \returns Human readable string containing useful device information
*/
std::string Device::getStats()
    {
    std::string s("OptiX devices:");

    vector<int> devices = m_context->getEnabledDevices();

    for (const int& i : devices)
        {
        s += "\n[" + to_string(i) + "]: " + m_context->getDeviceName(i);
        }

    return s;
    }

/*! \param m Python module to export in
 */
void export_Device(pybind11::module& m)
    {
    pybind11::class_<Device, std::shared_ptr<Device> >(m, "Device")
        .def(pybind11::init<>())
        .def("getStats", &Device::getStats)
        ;
    }

} } // end namespace fresnel::gpu
