// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerWhitted.h"
#include <cmath>

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerWhitted::TracerWhitted(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    }

TracerWhitted::~TracerWhitted()
    {
    }

void TracerWhitted::render(std::shared_ptr<Scene> scene)
    {
    Tracer::render(scene);
    rtcCommit(scene->getRTCScene());
    m_device->checkError();

    for (unsigned int j = 0; j < m_h; j++)
        {
        for (unsigned int i = 0; i < m_w; i++)
            {
            float y = j/float(m_h-1)*2-1;
            float x = i/float(m_w-1)*2-1;

            RTCRay ray(vec3<float>(x,y,-1), vec3<float>(0,0,1));
            rtcIntersect(scene->getRTCScene(), ray);

            float c = 0.0;
            float a = 0.0;
            if (ray.hit())
                {
                ray.Ng = ray.Ng / std::sqrt(dot(ray.Ng, ray.Ng));
                c = dot(ray.Ng, vec3<float>(-1,-1,0));
                a = 1.0;
                }

            unsigned int pixel = j*m_w + i;
            m_out[pixel].r = c;
            m_out[pixel].g = c;
            m_out[pixel].b = c;
            m_out[pixel].a = a;
            }
        }
    }

/*! \param m Python module to export in
 */
void export_TracerWhitted(pybind11::module& m)
    {
    pybind11::class_<TracerWhitted, std::shared_ptr<TracerWhitted> >(m, "TracerWhitted", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def_buffer([](TracerWhitted &t) -> pybind11::buffer_info { return t.getBuffer(); })  // repeated because def_buffer does not inherit
        ;
    }

} } // end namespace fresnel::cpu
