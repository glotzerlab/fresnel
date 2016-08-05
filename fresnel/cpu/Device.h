// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

namespace fresnel { namespace cpu {

// Thin wrapper for RTCDevice
/* Handle construction and deletion of the device, and python lifetime as an exported class.
*/
class Device
    {
    public:
        //! Constructor
        Device();

        //! Destructor
        ~Device();

        //! Access the RTCDevice
        RTCDevice& getRTCDevice()
            {
            return m_device;
            }
    private:
        RTCDevice m_device; //!< Store the device
    };

//! Export Device to python
void export_Device(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
