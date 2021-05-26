// Copyright (c) 2016-2021 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <memory>
#include <sstream>
#include <stdexcept>

#include "Device.h"

namespace fresnel
    {
namespace cpu
    {
/*! Construct a new RTCDevice with default options.
    \param limit Set a maximum number of threads limit. -1 will auto-select.
*/
Device::Device(int limit) : m_limit(limit)
    {
    if (limit == -1)
        {
        m_device = rtcNewDevice(nullptr);
        limit = tbb::task_arena::automatic;
        }
    else
        {
        std::ostringstream config;
        config << "threads=" << limit;
        m_device = rtcNewDevice(config.str().c_str());
        }

    m_arena = std::make_shared<tbb::task_arena>(limit);

    if (m_device == nullptr)
        {
        throw std::runtime_error("Error creating embree device");
        }
    }

/*! Destroy the underlying RTCDevice
 */
Device::~Device()
    {
    rtcReleaseDevice(m_device);
    }

/*! \param m Python module to export in
 */
void export_Device(pybind11::module& m)
    {
    pybind11::class_<Device, std::shared_ptr<Device>>(m, "Device")
        .def(pybind11::init<int>())
        .def("describe", &Device::describe);
    }

    } // namespace cpu
    } // namespace fresnel
