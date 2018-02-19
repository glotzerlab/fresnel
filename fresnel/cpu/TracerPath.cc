// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerPath.h"
#include "common/RayGen.h"
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
                // create the ray generator for this pixel
                RayGen ray_gen(i, j, width, height, m_seed);

                // determine the viewing plane relative coordinates of this pixel
                vec2<float> sample_loc = ray_gen.importanceSampleAA(m_n_samples);

                vec3<float> origin = cam.origin(sample_loc);
                vec3<float> direction = cam.direction(sample_loc);

                // determine the output pixel color
                RGB<float> c(0,0,0);
                float a = 1.0f;

                // trace the first ray into the scene
                RTCRay ray_initial(origin,  direction);
                ray_initial.tnear = 1e-3f;
                rtcIntersect(scene->getRTCScene(), ray_initial);

                if (!ray_initial.hit())
                    {
                    // no need to resample the backround N times
                    c = background_color * float(m_light_samples);
                    a = background_alpha;
                    }
                else
                    {
                    // trace a path from the hit point into the scene m_light_samples times
                    for (unsigned int light_sample = 0; light_sample < m_light_samples; light_sample++)
                        {
                        RGB<float> attenuation(1.0f,1.0f,1.0f);
                        for (unsigned int depth = 1; depth <= 10000; depth++)
                            {
                            RTCRay ray(origin,  direction);
                            if (depth == 1)
                                {
                                // the first hit is cached above
                                ray = ray_initial;
                                }
                            else
                                {
                                // subsequent depth steps need to trace
                                ray.tnear = 1e-3f;
                                rtcIntersect(scene->getRTCScene(), ray);
                                }

                            if (ray.hit())
                                {
                                vec3<float> n = ray.Ng / std::sqrt(dot(ray.Ng, ray.Ng));
                                vec3<float> v = -ray.dir;
                                Material m;

                                // apply the material color or outline color depending on the distance to the edge
                                if (ray.d > scene->getOutlineWidth(ray.geomID))
                                    m = scene->getMaterial(ray.geomID);
                                else
                                    m = scene->getOutlineMaterial(ray.geomID);

                                if (m.isSolid())
                                    {
                                    // testing: treat solid colors as emitters
                                    // when the ray hits an emitter, terminate the path and evaluate the final color
                                    c += attenuation * m.getColor(ray.shading_color);
                                    break;
                                    }
                                else
                                    {
                                    // when the ray hits an object with a normal material, update the attenuation using the BRDF
                                    // choose a random direction l to continue the path.
                                    float factor = 1.0;
                                    bool transmit = false;

                                    vec3<float> l = ray_gen.MISReflectionTransmission(factor,
                                                                                      transmit,
                                                                                      v,
                                                                                      n,
                                                                                      depth,
                                                                                      (m_n_samples-1)*m_light_samples + light_sample,
                                                                                      m);
                                    if (transmit)
                                        {
                                        // perfect transmission
                                        RGB<float> trans_color = m.getColor(ray.shading_color);
                                        trans_color.r = sqrtf(trans_color.r);
                                        trans_color.g = sqrtf(trans_color.g);
                                        trans_color.b = sqrtf(trans_color.b);
                                        attenuation *= trans_color;
                                        }
                                    else
                                        {
                                        float ndotl = dot(n, l);

                                        // When n dot v is less than 0, this is coming through the back of the surface
                                        // skip this sample
                                        if (dot(n,v) > 0.0f && ndotl > 0.0f)
                                            {
                                            attenuation *= m.brdf(l, v, n, ray.shading_color) *
                                                           ndotl *
                                                           factor;
                                            }
                                        else
                                            {
                                            attenuation = RGB<float>(0,0,0);
                                            break;
                                            }
                                        }

                                    // set the origin and direction for the next ray in the path
                                    origin = ray.org + ray.dir * ray.tfar;
                                    direction = l;
                                    }
                                }
                            else
                                {
                                // ray missed geometry entirely (and depth > 1)
                                // see if it hit any lights and compute the output color accordingly
                                for (unsigned int light_id = 0; light_id < lights.N; light_id++)
                                    {
                                    vec3<float> l = lights.direction[light_id];
                                    float half_angle = lights.theta[light_id];
                                    float cos_half_angle = cosf(half_angle);
                                    if (half_angle > float(M_PI)/2.0f)
                                        half_angle = float(M_PI)/2.0f;
                                    float ldotd = dot(l,ray.dir);

                                    // NB: we can do this because ray directions are always normalized
                                    if (ldotd >= cos_half_angle)
                                        {
                                        // hit the light
                                        // the division bin sin(light half angle) normalizes the lights so that
                                        // a light of color 1 of any non-zero size results in an output of 1
                                        // when that light is straight over the surface
                                        c += attenuation * lights.color[light_id] / sinf(half_angle);
                                        }
                                    } // end loop over lights

                                // the ray missed geometry, no more tracing to do
                                break;
                                }
                            } // end depth loop
                        } // end light samples loop
                    } // end else block of if initial ray did not hit

                // take the current sample and compute the average with the previous samples
                RGBA<float> output_sample(c / float(m_light_samples), a);

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
                    srgb_output[pixel] = sRGB(RGBA<float>(m_highlight_warning_color, a));
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
