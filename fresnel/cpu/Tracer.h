// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_H_
#define TRACER_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "common/ColorMath.h"
#include "common/Camera.h"
#include "Scene.h"
#include "Array.h"

namespace fresnel { namespace cpu {

//! Base class for the raytracer
/*! The base class Tracer specifies common methods used for all tracers. This includes output buffer management,
    and defining the rendering API.

    The output buffer is stored in two formats. *m_linear_out* store the ray traced output in a linear RGB color space.
    This buffer is suitable for tone mapping and averaging with other render output. *m_srgb_out* stores the output
    in the sRGB color space and in a 4 bytes per pixel format suitable for direct use in image display.

    The rendering methods themselves do nothing. Derived classes must implement them.
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

        //! Get the SRGB output pixel buffer
        virtual std::shared_ptr< Array< RGBA<unsigned char> > > getSRGBOutputBuffer()
            {
            return m_srgb_out;
            }

        //! Get the SRGB output pixel buffer
        virtual std::shared_ptr< Array< RGBA<float> > > getLinearOutputBuffer()
            {
            return m_linear_out;
            }

    protected:
        std::shared_ptr<Device> m_device;                           //!< The device the Scene is attached to
        std::shared_ptr< Array< RGBA<float> > > m_linear_out;       //!< The output buffer (linear space)
        std::shared_ptr< Array< RGBA<unsigned char> > > m_srgb_out; //!< The output buffer (srgb space)
    };

//! Export Tracer to python
void export_Tracer(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
