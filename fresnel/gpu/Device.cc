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
    RTresult result;

    result = rtContextCreate(&m_context);
    checkError(result);
    }

/*! Destroy the underlying RTcontext
*/
Device::~Device()
    {
    std::cout << "Destroy GPU Device" << std::endl;
    RTresult result;

    result = rtContextDestroy(m_context);
    checkError(result);
    }

/*! \returns Human readable string containing useful device information
*/
std::string Device::getStats()
    {
    unsigned int n;
    RTresult result;
    std::string s("OptiX devices:");

    result = rtContextGetDeviceCount(m_context, &n);
    checkError(result);

    vector<int> devices(n);
    result = rtContextGetDevices(m_context, &devices[0]);

    for (const int& i : devices)
        {
        char name[120];
        result = rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, 120, name);
        checkError(result);
        s += "\n[" + to_string(i) + "]: " + string(name);
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
