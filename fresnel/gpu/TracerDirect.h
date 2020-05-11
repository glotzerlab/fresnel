// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_WHITTED_H_
#define TRACER_WHITTED_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel
    {
namespace gpu
    {
//! Basic Direct ray tracer using OptiX
/*! GPU code for the tracer is in direct.cu

    TracerDirect loads the direct ray generation program and adds it to the Device.
*/
class TracerDirect : public Tracer
    {
    public:
    //! Constructor
    TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h);
    //! Destructor
    virtual ~TracerDirect();

    //! Initialize the Material for use in tracing
    static void setupMaterial(optix::Material mat, Device* dev);

    //! Render a scene
    void render(std::shared_ptr<Scene> scene);

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

    } // namespace gpu
    } // namespace fresnel

#endif
