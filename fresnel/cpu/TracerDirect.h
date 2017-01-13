// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_WHITTED_H_
#define TRACER_WHITTED_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel { namespace cpu {

//! Basic Direct raytracer
/*!
*/
class TracerDirect : public Tracer
    {
    public:
        //! Constructor
        TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h);
        //! Destructor
        virtual ~TracerDirect();

        //! Render a scene
        virtual void render(std::shared_ptr<Scene> scene);
    };

//! Export TracerDirect to python
void export_TracerDirect(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
