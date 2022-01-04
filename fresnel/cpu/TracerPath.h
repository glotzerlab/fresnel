// Copyright (c) 2016-2022 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef TRACER_PATH_H_
#define TRACER_PATH_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Tracer.h"

namespace fresnel
    {
namespace cpu
    {
//! Path tracer
/*! The path tracer randomly samples light paths in the scene to obtain soft lighting from area
   light sources and other global illumination techniques (reflection, refraction, anti-aliasing,
   etc...).

    Every time render() is called, a sample is taken and the output updated to match the current
   average. Many samples may be needed to obtain a converged image. Call reset() to clear the
   current image and start a new sampling run. The Tracer does not know when the camera angle,
   materials, or other properties of the scene have changed, so the caller must call reset()
   whenever needed to start sampling a new view or changed scene (unless motion blur or other
   multiple exposure techniques are the desired output).
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

    //! Render a scene
    virtual void render(std::shared_ptr<Scene> scene);

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

    /// Implementation of the render loop
    virtual void renderImplementation(std::shared_ptr<Scene> scene);
    };

//! Export TracerDirect to python
void export_TracerPath(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
