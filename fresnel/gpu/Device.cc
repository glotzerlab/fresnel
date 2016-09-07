// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <memory>
#include <string>

#include "Device.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! Construct a new optix::Context with default options.

    \param ptx_root Directory where the PTX files are stored

    OptiX programs are loaded from PTX files, built from the .cu source files. These PTX files are stored in the
    python library directory. The Device instance tracks this directory for other classes (i.e. Tracer) to use
    when loading OptiX programs.
*/
Device::Device(const std::string& ptx_root) : m_ptx_root(ptx_root)
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
        .def(pybind11::init<const std::string&>())
        .def("getStats", &Device::getStats)
        ;
    }

} } // end namespace fresnel::gpu
