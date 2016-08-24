// Copyright (c) 2016 The Regents of the University of Michigan
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

namespace fresnel { namespace cpu {

//! Base class for the raytracer
/*! The base class Tracer specifies common methods used for all tracers. This includes output buffer management,
    defining the rendering API, camera specification, etc...

    The output buffer is stored in floating point rgba format in a unique pointer. For python access,
    Tracer specifies a buffer protocol so that numpy.array(tracer, copy=False) (or any other means of buffer
    access) can get at the data nocopy. However, a call to resize will invalidate the buffer.

    TODO: Consider a separate RGBA 4-byte per pixel output buffer to improve frame rates, separate from the
          render buffer
    TODO: Consider allocating the output buffer with a maximum size in mind so that resize will never invalidate
          numpy array handles and all handles will be valid indefinitely.

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

        //! Set the camera
        virtual void setCamera(const Camera& camera)
            {
            m_camera = camera;
            }

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

} } // end namespace fresnel::cpu

#endif
