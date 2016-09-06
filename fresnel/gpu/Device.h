// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include <optix_host.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

// setup pybind11 to use std::shared_ptr
PYBIND11_DECLARE_HOLDER_TYPE(T_shared_ptr_bind, std::shared_ptr<T_shared_ptr_bind>);

namespace fresnel { namespace gpu {

//! Thin wrapper for RTContext
/* Handle construction and deletion of the optix context, and python lifetime as an exported class.
*/
class Device
    {
    public:
        //! Constructor
        Device();

        //! Destructor
        ~Device();

        //! Access the RTCDevice
        RTcontext& getRTcontext()
            {
            return m_context;
            }

        //! Check for error conditions
        /*! Throws an exception when an Embree error code is set
        */
        void checkError(const RTresult& err)
            {
            if (err == RT_SUCCESS)
                {
                return;
                }
            else
                {
                const char *err_str;
                rtContextGetErrorString(m_context, err, &err_str);
                throw std::runtime_error("Optix: " + std::string(err_str));
                }
            }

        //! Get information about this device
        std::string getStats();

    private:
        RTcontext m_context; //!< Store the context
    };

//! Export Device to python
void export_Device(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
