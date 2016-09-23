// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerWhitted.h"
#include <cmath>
#include <stdexcept>

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerWhitted::TracerWhitted(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    std::cout << "Create TracerWhitted" << std::endl;
    }

TracerWhitted::~TracerWhitted()
    {
    std::cout << "Destroy TracerWhitted" << std::endl;
    }

void TracerWhitted::render(std::shared_ptr<Scene> scene)
    {
    Material edge;
    edge.solid = 1.0;
    edge.color = RGB<float>(0,0,0);

    Camera cam = m_camera;
    Tracer::render(scene);

    // update Embree data structures
    rtcCommit(scene->getRTCScene());
    m_device->checkError();

    // for each pixel
    for (unsigned int j = 0; j < m_h; j++)
        {
        for (unsigned int i = 0; i < m_w; i++)
            {
            // determine the viewing plane relative coordinates
            float ys = -1.0f*(j/float(m_h-1)-0.5f);
            float xs = i/float(m_h-1)-0.5f*float(m_w)/float(m_h);

            // trace a ray into the scene
            RTCRay ray(cam.origin(xs, ys), cam.direction(xs, ys));
            rtcIntersect(scene->getRTCScene(), ray);

            // determine the output pixel color
            RGB<float> c(0,0,0);
            float a = 0.0;

            if (ray.hit())
                {
                vec3<float> n = ray.Ng / std::sqrt(dot(ray.Ng, ray.Ng));
                vec3<float> l = vec3<float>(1,1,1);
                l = l / sqrtf(dot(l,l));
                vec3<float> v = -ray.dir / std::sqrt(dot(ray.dir, ray.dir));
                Material m;

                // apply the material color or outline color depending on the distance to the edge
                if (ray.d > 0.15)
                    m = scene->getMaterial(ray.geomID);
                else
                    m = edge;

                if (m.isSolid())
                    {
                    c = m.color;
                    }
                else
                    {
                    // only apply brdf when the light faces the surface
                    float ndotl = dot(n,l);
                    if (ndotl > 0.0f)
                        {
                        c = m.brdf(l, v, n) * float(M_PI) * /* light color * */ ndotl;
                        }
                    }

                a = 1.0;
                }

            // write the output pixel
            unsigned int pixel = j*m_w + i;
            m_out[pixel].r = c.r;
            m_out[pixel].g = c.g;
            m_out[pixel].b = c.b;
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
