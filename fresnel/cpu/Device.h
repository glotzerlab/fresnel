// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include "embree_platform.h"
#include "tbb/task_arena.h"
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "Array.h" // not used by device, but for the pybind11 shared pointer holder type definition

namespace fresnel
    {
namespace cpu
    {
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
        RTCError err = rtcGetDeviceError(m_device);

        switch (err)
            {
        case RTC_ERROR_NONE:
            break;
        case RTC_ERROR_UNKNOWN:
            throw std::runtime_error("Embree: An unknown error has occurred.");
            break;
        case RTC_ERROR_INVALID_ARGUMENT:
            throw std::runtime_error("Embree: An invalid argument was specified.");
            break;
        case RTC_ERROR_INVALID_OPERATION:
            throw std::runtime_error(
                "Embree: The operation is not allowed for the specified object.");
            break;
        case RTC_ERROR_OUT_OF_MEMORY:
            throw std::runtime_error(
                "Embree: There is not enough memory left to complete the operation.");
            break;
        case RTC_ERROR_UNSUPPORTED_CPU:
            throw std::runtime_error(
                "Embree: The CPU is not supported as it does not support SSE2.");
            break;
        case RTC_ERROR_CANCELLED:
            throw std::runtime_error("Embree: The operation got cancelled by an Memory Monitor "
                                     "Callback or Progress Monitor Callback function.");
            break;
        default:
            throw std::runtime_error("Embree: An invalid error has occurred.");
            break;
            }
        }

    std::string describe()
        {
        if (m_limit == -1)
            return std::string("All available CPU threads");
        else
            return std::string("") + std::to_string(m_limit) + " CPU threads";
        }

    private:
    RTCDevice m_device; //!< Store the embree device
    std::shared_ptr<tbb::task_arena> m_arena; //!< TBB task arena
    int m_limit; //!< Cached limit for reporting to users
    };

//! Export Device to python
void export_Device(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
