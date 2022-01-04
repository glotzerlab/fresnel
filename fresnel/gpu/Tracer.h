// Copyright (c) 2016-2022 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef TRACER_H_
#define TRACER_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Array.h"
#include "Scene.h"
#include "common/Camera.h"
#include "common/ColorMath.h"

namespace fresnel
    {
namespace gpu
    {
//! Base class for the GPU ray tracer
/*! Provides common methods for ray tracing matching the API of cpu::Tracer. The base class does not
   render a scene, derived classes must implement rendering methods and load program objects. The
   base class does manage the output buffer and store the camera, width, height, and relevant OptiX
   program.

    Member variables derived classes must fill out to make a valid Tracer:

      - m_ray_gen is the ray generation program, loaded by derived classes
      - m_ray_gen_entry is the entry point index in the context for m_ray_gen.
*/
class Tracer
    {
    public:
    //! Constructor
    Tracer(std::shared_ptr<Device> device, unsigned int w, unsigned int h);

    //! Destructor
    virtual ~Tracer();

    //! Resize the output buffer
    virtual void resize(unsigned int w, unsigned int h);

    //! Render a scene
    virtual void render(std::shared_ptr<Scene> scene);

    //! Get the linear output pixel buffer
    virtual std::shared_ptr<Array<RGBA<float>>> getLinearOutputBuffer()
        {
        return m_linear_out_py;
        }

    //! Get the linear output pixel buffer
    virtual std::shared_ptr<Array<RGBA<unsigned char>>> getSRGBOutputBuffer()
        {
        return m_srgb_out_py;
        }

    //! Enable highlight warnings
    void enableHighlightWarning(const RGB<float>& color)
        {
        m_highlight_warning = true;
        m_highlight_warning_color = color;
        }

    void disableHighlightWarning()
        {
        m_highlight_warning = false;
        }

    //! Set the random number seed
    void setSeed(unsigned int seed)
        {
        m_seed = seed;
        }

    //! Get the random number seed
    unsigned int getSeed() const
        {
        return m_seed;
        }

    protected:
    std::shared_ptr<Device> m_device; //!< The device the Scene is attached to
    unsigned int m_w; //!< Width of the output buffer
    unsigned int m_h; //!< Height of the output buffer
    optix::Buffer m_linear_out_gpu; //!< The GPU linear output buffer
    optix::Buffer m_srgb_out_gpu; //!< The GPU linear output buffer

    std::shared_ptr<Array<RGBA<float>>> m_linear_out_py; //!< The linear output buffer for python
    std::shared_ptr<Array<RGBA<unsigned char>>>
        m_srgb_out_py; //!< The sRGB output buffer for python

    optix::Program m_ray_gen; //!< Ray generation program
    optix::Program m_exception_program; //!< Exception program
    unsigned int m_ray_gen_entry; //!< Entry point of the ray generation program

    bool m_highlight_warning; //!< Set to true to enable highlight warnings in sRGB output
    RGB<float> m_highlight_warning_color; //!< The highlight warning color
    unsigned int m_seed = 0; //!< Random number seed
    };

//! Export Tracer to python
void export_Tracer(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
