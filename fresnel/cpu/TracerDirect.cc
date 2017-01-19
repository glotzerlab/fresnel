// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerDirect.h"
#include <cmath>
#include <stdexcept>

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    std::cout << "Create TracerDirect" << std::endl;
    }

TracerDirect::~TracerDirect()
    {
    std::cout << "Destroy TracerDirect" << std::endl;
    }

void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();
    const vec3<float> light_direction = scene->getLightDirection();

    const Camera cam = scene->getCamera();
    Tracer::render(scene);

    // update Embree data structures
    rtcCommit(scene->getRTCScene());
    m_device->checkError();

    RGBA<float>* linear_output = m_linear_out->map();
    RGBA<unsigned char>* srgb_output = m_srgb_out->map();

    // for each pixel
    const unsigned int height = m_linear_out->getH();
    const unsigned int width = m_linear_out->getW();
    for (unsigned int j = 0; j < height; j++)
        {
        for (unsigned int i = 0; i < width; i++)
            {
            // determine the viewing plane relative coordinates
            float ys = -1.0f*(j/float(height-1)-0.5f);
            float xs = i/float(height-1)-0.5f*float(width)/float(height);

            // trace a ray into the scene
            RTCRay ray(cam.origin(xs, ys), cam.direction(xs, ys));
            rtcIntersect(scene->getRTCScene(), ray);

            // determine the output pixel color
            RGB<float> c = background_color;
            float a = background_alpha;

            if (ray.hit())
                {
                vec3<float> n = ray.Ng / std::sqrt(dot(ray.Ng, ray.Ng));
                vec3<float> l = light_direction;
                vec3<float> v = -ray.dir / std::sqrt(dot(ray.dir, ray.dir));
                Material m;

                // apply the material color or outline color depending on the distance to the edge
                if (ray.d > scene->getOutlineWidth(ray.geomID))
                    m = scene->getMaterial(ray.geomID);
                else
                    m = scene->getOutlineMaterial(ray.geomID);

                if (m.isSolid())
                    {
                    c = m.getColor(ray.shading_color);
                    }
                else
                    {
                    // only apply brdf when the light faces the surface
                    float ndotl = dot(n,l);
                    if (ndotl > 0.0f)
                        {
                        c = m.brdf(l, v, n, ray.shading_color) * float(M_PI) * /* light color * */ ndotl;
                        }
                    else
                        {
                        c = RGB<float>(0,0,0);
                        }
                    }

                a = 1.0;
                }

            // write the output pixel
            unsigned int pixel = j*width + i;
            RGBA<float> output_pixel(c.r, c.g, c.b, a);
            linear_output[pixel] = output_pixel;
            srgb_output[pixel] = sRGB(output_pixel);
            }
        }

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, std::shared_ptr<TracerDirect> >(m, "TracerDirect", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        ;
    }

} } // end namespace fresnel::cpu
