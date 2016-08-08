// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerWhitted.h"

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

            RTCRay ray;
            ray.org[0] = x; ray.org[1] = y; ray.org[2] = -1;
            ray.dir[0] = ray.dir[1] = 0; ray.dir[2] = 1;
            ray.tnear = 0.0f;
            ray.tfar = std::numeric_limits<float>::infinity();
            ray.instID = RTC_INVALID_GEOMETRY_ID;
            ray.geomID = RTC_INVALID_GEOMETRY_ID;
            ray.primID = RTC_INVALID_GEOMETRY_ID;
            ray.mask = 0xFFFFFFFF;
            ray.time = 0.0f;
            rtcIntersect(scene->getRTCScene(), ray);

            float c = 0.0;
            if (ray.geomID != RTC_INVALID_GEOMETRY_ID)
                c = 1.0;


            unsigned int pixel = j*m_w + i;
            m_out[pixel].r = c;
            m_out[pixel].g = c;
            m_out[pixel].b = c;
            m_out[pixel].a = 1.0;
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
