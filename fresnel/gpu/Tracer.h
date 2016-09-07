// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef TRACER_H_
#define TRACER_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "common/ColorMath.h"
#include "common/Camera.h"
#include "Scene.h"

namespace fresnel { namespace gpu {

//! Base class for the GPU ray tracer
/*! Provides common methods for ray tracing matching the API of cpu::Tracer. The base class does not render a scene,
    derived classes must implement rendering methods.
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

        //! Set the camera
        virtual void setCamera(const Camera& camera);

        //! Get a python buffer pointing to the pixel output buffer
        virtual pybind11::buffer_info getBuffer();

    protected:
        std::shared_ptr<Device> m_device;  //!< The device the Scene is attached to
        unsigned int m_w;                  //!< Width of the output buffer
        unsigned int m_h;                  //!< Height of the output buffer
        std::unique_ptr<RGBA<float>[]> m_out;    //!< The output buffer

        Camera m_camera;                   //!< The camera
    };

//! Export Tracer to python
void export_Tracer(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
