// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_WHITTED_H_
#define TRACER_WHITTED_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel { namespace gpu {

//! Basic Whitted ray tracer using OptiX
/*! GPU code for the tracer is in whitted.cu

    TracerWhitted loads the whitted ray generation program and adds it to the Device.
*/
class TracerWhitted : public Tracer
    {
    public:
        //! Constructor
        TracerWhitted(std::shared_ptr<Device> device, unsigned int w, unsigned int h);
        //! Destructor
        virtual ~TracerWhitted();
    };

//! Export TracerWhitted to python
void export_TracerWhitted(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
