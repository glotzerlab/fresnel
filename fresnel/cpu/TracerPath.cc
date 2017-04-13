// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerPath.h"
#include <cmath>
#include <stdexcept>

#include "tbb/tbb.h"

using namespace tbb;

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerPath::TracerPath(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    }

TracerPath::~TracerPath()
    {
    }

void TracerPath::reset()
    {
    m_n_samples = 0;
    RGBA<float>* linear_output = m_linear_out->map();
    memset(linear_output, 0, sizeof(RGBA<float>)*m_linear_out->getW()*m_linear_out->getH());
    m_linear_out->unmap();

    RGBA<unsigned char>* srgb_output = m_srgb_out->map();
    memset(srgb_output, 0, sizeof(RGBA<unsigned char>)*m_linear_out->getW()*m_linear_out->getH());
    m_srgb_out->unmap();
    }

void TracerPath::render(std::shared_ptr<Scene> scene)
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

    // update number of samples (the first sample is 1)
    m_n_samples++;

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

                        // choose a random direction to

                        for (unsigned int light_id = 0; light_id < lights.N; light_id++)
                            {
                            vec3<float> l = lights.direction[light_id];

                            // find the representative point, a vector pointing to the a point on the area light
                            // with a smallest angle to the reflection vector
                            vec3<float> r = -v + (2.0f * n * dot(n,v));

                            // find the closest point on the area light
                            float half_angle = lights.theta[light_id];
                            float cos_half_angle = cosf(half_angle);
                            float ldotr = dot(l,r);
                            if (ldotr < cos_half_angle)
                                {
                                vec3<float> a = cross(l,r);
                                a = a / sqrtf(dot(a,a));

                                // miss the light, need to rotate r by the difference in the angles about l cross r
                                quat<float> q = quat<float>::fromAxisAngle(a, -acosf(ldotr) + half_angle);
                                r = rotate(q, r);
                                }
                            else
                                {
                                // hit the light, no modification necessary to r
                                }

                            // only apply brdf when the light faces the surface
                            RGB<float> f_d;
                            float ndotl = dot(n, l);
                            if (ndotl >= 0.0f)
                                f_d = m.brdf_diffuse(l, v, n, ray.shading_color) * ndotl;
                            else
                                f_d = RGB<float>(0.0f,0.0f,0.0f);

                            RGB<float> f_s;
                            if (dot(n, r) >= 0.0f)
                                {
                                f_s = m.brdf_specular(r, v, n, ray.shading_color, half_angle) * dot(n, r);
                                }
                            else
                                f_s = RGB<float>(0.0f,0.0f,0.0f);

                            c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
                            }
                        }

                    a = 1.0;
                    }

                // take the current sample and compute the average with the previous samples
                unsigned int pixel = j*width + i;
                RGBA<float> output_sample(c, a);

                // running average and variance (TODO) using Welford's method
                // (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                RGBA<float> old_mean = linear_output[pixel];
                linear_output[pixel] = old_mean + (output_sample-old_mean)/float(m_n_samples);

                // convert the current average output to sRGB
                RGBA<float> output_pixel = linear_output[pixel];
                if (!m_highlight_warning || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
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
void export_TracerPath(pybind11::module& m)
    {
    pybind11::class_<TracerPath, std::shared_ptr<TracerPath> >(m, "TracerPath", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("getNumSamples", &TracerPath::getNumSamples)
        .def("reset", &TracerPath::reset)
        ;
    }

} } // end namespace fresnel::cpu
