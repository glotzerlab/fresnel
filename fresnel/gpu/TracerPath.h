// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef TRACER_PATH_H_
#define TRACER_PATH_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel
    {
namespace gpu
    {
//! Path tracer
/*! GPU code for the tracer is in path.cu

    See cpu::TracerPath for API documentation
*/
class TracerPath : public Tracer
    {
    public:
    //! Constructor
    TracerPath(std::shared_ptr<Device> device,
               unsigned int w,
               unsigned int h,
               unsigned int light_samples);
    //! Destructor
    virtual ~TracerPath();

    //! Initialize the Material for use in tracing
    static void setupMaterial(optix::Material mat, Device* dev);

    //! Render a scene
    void render(std::shared_ptr<Scene> scene);

    //! Reset the sampling
    virtual void reset();

    //! Resize the output buffer
    virtual void resize(unsigned int w, unsigned int h)
        {
        Tracer::resize(w, h);
        m_n_samples = 0;
        m_seed++;
        }

    //! Get the number of samples taken
    unsigned int getNumSamples() const
        {
        return m_n_samples;
        }

    //! Set the number of light samples
    void setLightSamples(unsigned int light_samples)
        {
        m_light_samples = light_samples;
        }

    protected:
    unsigned int m_n_samples; //!< Number of samples taken since the last reset
    unsigned int m_light_samples; //!< Number of light samples to take each render()
    };

//! Export TracerPath to python
void export_TracerPath(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
