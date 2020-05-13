// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_WHITTED_H_
#define TRACER_WHITTED_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel
    {
namespace cpu
    {
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

    //! Set the number of AA samples in each direction
    void setAntialiasingN(unsigned int n)
        {
        m_aa_n = n;
        }

    //! Get the number of AA samples in each direction
    unsigned int getAntialiasingN() const
        {
        return m_aa_n;
        }

    protected:
    //! Number of AA samples in each direction
    unsigned int m_aa_n = 8;
    };

//! Export TracerDirect to python
void export_TracerDirect(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
