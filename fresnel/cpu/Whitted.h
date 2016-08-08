// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef WHITTED_H_
#define WHITTED_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Scene.h"

namespace fresnel { namespace cpu {

//! Basic Whitted raytracer
/*!
*/
class Whitted
    {
    public:
        //! Constructor
        Whitted(std::shared_ptr<Device> device);
        //! Destructor
        virtual ~Whitted();

        //! Render a scene
        virtual void render(std::shared_ptr<Scene> scene, unsigned int w, unsigned int h);
    protected:
        std::shared_ptr<Device> m_device;  //!< The device the Scene is attached to
    };

//! Export Whitted to python
void export_Whitted(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
