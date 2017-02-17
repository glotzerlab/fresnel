// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "tbb/task_arena.h"

#include "Array.h"      // not used by device, but for the pybind11 shared pointer holder type definition

namespace fresnel { namespace cpu {

//! Thin wrapper for RTCDevice
/* Handle construction and deletion of the device, and python lifetime as an exported class.
*/
class Device
    {
    public:
        //! Constructor
        Device(int limit);

        //! Destructor
        ~Device();

        //! Access the RTCDevice
        RTCDevice& getRTCDevice()
            {
            return m_device;
            }

        //! Get the TBB task arena
        std::shared_ptr<tbb::task_arena> getTBBArena()
            {
            return m_arena;
            }

        //! Check for error conditions
        /*! Throws an exception when an Embree error code is set
        */
        void checkError()
            {
            RTCError err = rtcDeviceGetError(m_device);

            switch (err)
                {
                case RTC_NO_ERROR:
                    break;
                case RTC_UNKNOWN_ERROR:
                    throw std::runtime_error("Embree: An unknown error has occurred.");
                    break;
                case RTC_INVALID_ARGUMENT:
                    throw std::runtime_error("Embree: An invalid argument was specified.");
                    break;
                case RTC_INVALID_OPERATION:
                    throw std::runtime_error("Embree: The operation is not allowed for the specified object.");
                    break;
                case RTC_OUT_OF_MEMORY:
                    throw std::runtime_error("Embree: There is not enough memory left to complete the operation.");
                    break;
                case RTC_UNSUPPORTED_CPU:
                    throw std::runtime_error("Embree: The CPU is not supported as it does not support SSE2.");
                    break;
                case RTC_CANCELLED:
                    throw std::runtime_error("Embree: The operation got cancelled by an Memory Monitor Callback or Progress Monitor Callback function.");
                    break;
                default:
                    throw std::runtime_error("Embree: An invalid error has occurred.");
                    break;
                }
            }

    private:
        RTCDevice m_device;                         //!< Store the embree device
        std::shared_ptr<tbb::task_arena> m_arena;   //!< TBB task arena
    };

//! Export Device to python
void export_Device(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
