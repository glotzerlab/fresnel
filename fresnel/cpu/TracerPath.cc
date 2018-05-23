// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerPath.h"
#include "common/RayGen.h"
#include "common/TracerPathMethods.h"
#include <cmath>
#include <stdexcept>

#include "tbb/tbb.h"

using namespace tbb;

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
TracerPath::TracerPath(std::shared_ptr<Device> device, unsigned int w, unsigned int h, unsigned int light_samples)
    : Tracer(device, w, h), m_light_samples(light_samples)
    {
    reset();
    }

TracerPath::~TracerPath()
    {
    }

void TracerPath::reset()
    {
    m_n_samples = 0;
    m_seed++;

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
    arena->execute([&]{ rtcCommitScene(scene->getRTCScene()); });
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
                // create the ray generator for this pixel
                RayGen ray_gen(i, j, width, height, m_seed);

                // determine the viewing plane relative coordinates of this pixel
                vec2<float> sample_loc = ray_gen.importanceSampleAA(m_n_samples);

                // per ray data
                PRDpath prd;
                prd.result = RGB<float>(0,0,0);
                prd.a = 1.0f;

                // trace the first ray into the scene
                RTCRayHit ray_hit_initial;
                RTCRay& ray_initial = ray_hit_initial.ray;

                const vec3<float>& org = cam.origin(sample_loc);
                ray_initial.org_x = org.x;
                ray_initial.org_y = org.y;
                ray_initial.org_z = org.z;

                const vec3<float>& dir = cam.direction(sample_loc);
                ray_initial.dir_x = dir.x;
                ray_initial.dir_y = dir.y;
                ray_initial.dir_z = dir.z;

                ray_initial.tnear = 1e-3f;
                ray_initial.tfar = std::numeric_limits<float>::infinity();
                ray_initial.time = 0.0f;
                ray_initial.mask = -1;
                ray_initial.flags = 0;
                ray_hit_initial.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                ray_hit_initial.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                FresnelRTCIntersectContext context;
                rtcInitIntersectContext(&context.context);

                rtcIntersect1(scene->getRTCScene(), &context.context, &ray_hit_initial);

                // trace a path from the hit point into the scene m_light_samples times
                for (prd.light_sample = 0; prd.light_sample < m_light_samples; prd.light_sample++)
                    {
                    prd.attenuation = RGB<float>(1.0f,1.0f,1.0f);
                    prd.done = false;

                    for (prd.depth = 0; ; prd.depth++)
                        {
                        RTCRayHit ray_hit;
                        if (prd.depth == 0)
                            {
                            // the first hit is cached above
                            ray_hit = ray_hit_initial;
                            }
                        else
                            {
                            RTCRay& ray = ray_hit.ray;
                            ray.org_x = prd.origin.x;
                            ray.org_y = prd.origin.y;
                            ray.org_z = prd.origin.z;

                            ray.dir_x = prd.direction.x;
                            ray.dir_y = prd.direction.y;
                            ray.dir_z = prd.direction.z;

                            ray.tnear = 1e-3f;
                            ray.tfar = std::numeric_limits<float>::infinity();
                            ray.time = 0.0f;
                            ray.mask = -1;
                            ray.flags = 0;
                            ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                            ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                            // subsequent depth steps need to trace
                            context = FresnelRTCIntersectContext();
                            rtcInitIntersectContext(&context.context);

                            rtcIntersect1(scene->getRTCScene(), &context.context, &ray_hit);
                            }

                        if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                            {
                            // call hit program
                            path_tracer_hit(prd,
                                            scene->getMaterial(ray_hit.hit.geomID),
                                            scene->getOutlineMaterial(ray_hit.hit.geomID),
                                            context.d,
                                            scene->getOutlineWidth(ray_hit.hit.geomID),
                                            context.shading_color,
                                            vec3<float>(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z),
                                            vec3<float>(ray_hit.ray.org_x, ray_hit.ray.org_y, ray_hit.ray.org_z),
                                            vec3<float>(ray_hit.ray.dir_x, ray_hit.ray.dir_y, ray_hit.ray.dir_z),
                                            ray_hit.ray.tfar,
                                            ray_gen,
                                            m_n_samples,
                                            m_light_samples);
                            }
                        else
                            {
                            // call miss program
                            path_tracer_miss(prd,
                                             background_color,
                                             background_alpha,
                                             m_light_samples,
                                             lights,
                                             vec3<float>(ray_hit.ray.dir_x, ray_hit.ray.dir_y, ray_hit.ray.dir_z)
                                             );
                            }

                        // break out of the loop when done
                        if (prd.done)
                            break;
                        } // end depth loop
                    } // end light samples loop

                // take the current sample and compute the average with the previous samples
                RGBA<float> output_sample(prd.result / float(m_light_samples), prd.a);

                // running average using Welford's method. Variance is not particularly useful to users,
                // so don't compute that.
                // (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                unsigned int pixel = j*width + i;
                RGBA<float> old_mean = linear_output[pixel];
                linear_output[pixel] = old_mean + (output_sample-old_mean)/float(m_n_samples);

                // convert the current average output to sRGB
                RGBA<float> output_pixel = linear_output[pixel];
                if (!m_highlight_warning || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
                    srgb_output[pixel] = sRGB(output_pixel);
                else
                    srgb_output[pixel] = sRGB(RGBA<float>(m_highlight_warning_color, output_pixel.a));
                } // end loop over pixels in a tile
            } // end loop over tiles in this work unit
        }); // end parallel loop over all tiles
    }); // end arena limited execution

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerPath(pybind11::module& m)
    {
    pybind11::class_<TracerPath, std::shared_ptr<TracerPath> >(m, "TracerPath", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int, unsigned int>())
        .def("getNumSamples", &TracerPath::getNumSamples)
        .def("reset", &TracerPath::reset)
        .def("setLightSamples", &TracerPath::setLightSamples)
        ;
    }

} } // end namespace fresnel::cpu
