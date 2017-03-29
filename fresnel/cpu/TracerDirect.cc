// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerDirect.h"
#include <cmath>
#include <stdexcept>

#include "tbb/tbb.h"

using namespace tbb;

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    }

TracerDirect::~TracerDirect()
    {
    }

void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    std::shared_ptr<tbb::task_arena> arena = scene->getDevice()->getTBBArena();

    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    const Camera cam(scene->getCamera());
    const Lights lights(scene->getLights(), cam);
    Tracer::render(scene);

    // update Embree data structures
    arena->execute([&]{ rtcCommit(scene->getRTCScene()); });
    m_device->checkError();

    RGBA<float>* linear_output = m_linear_out->map();
    RGBA<unsigned char>* srgb_output = m_srgb_out->map();

    // for each pixel
    const unsigned int height = m_linear_out->getH();
    const unsigned int width = m_linear_out->getW();

    const unsigned int TILE_SIZE_X = 4;
    const unsigned int TILE_SIZE_Y = 4;
    const unsigned int numTilesX = (width +TILE_SIZE_X-1)/TILE_SIZE_X;
    const unsigned int numTilesY = (height+TILE_SIZE_Y-1)/TILE_SIZE_Y;

    arena->execute([&]{
    parallel_for( blocked_range<size_t>(0,numTilesX*numTilesY), [=](const blocked_range<size_t>& r)
        {
        for(size_t tile = r.begin(); tile != r.end(); ++tile)
            {
            const unsigned int tileY = tile / numTilesX;
            const unsigned int tileX = tile - tileY * numTilesX;
            const unsigned int x0 = tileX * TILE_SIZE_X;
            const unsigned int x1 = std::min(x0+TILE_SIZE_X,width);
            const unsigned int y0 = tileY * TILE_SIZE_Y;
            const unsigned int y1 = std::min(y0+TILE_SIZE_Y,height);

            for (unsigned int j=y0; j<y1; j++) for (unsigned int i=x0; i<x1; i++)
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
                        c = RGB<float>(0,0,0);
                        for (unsigned int light_id = 0; light_id < lights.N; light_id++)
                            {
                            vec3<float> l = lights.direction[light_id];

                            // only apply brdf when the light faces the surface
                            float ndotl = dot(n,l);
                            if (ndotl > 0.0f)
                                {
                                c += m.brdf(l, v, n, ray.shading_color) * float(M_PI) * lights.color[light_id] * ndotl;
                                }
                            }
                        }

                    a = 1.0;
                    }

                // write the output pixel
                unsigned int pixel = j*width + i;
                RGBA<float> output_pixel(c, a);
                linear_output[pixel] = output_pixel;
                if (!m_highlight_warning || (c.r <= 1.0f && c.g <= 1.0f && c.b <= 1.0f))
                    srgb_output[pixel] = sRGB(output_pixel);
                else
                    srgb_output[pixel] = sRGB(RGBA<float>(m_highlight_warning_color, a));
                }
            }
        });
    });

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
