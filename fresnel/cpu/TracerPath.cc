// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerPath.h"
#include <cmath>
#include <stdexcept>

#include "tbb/tbb.h"
#include "Random123/philox.h"
#include "boxmuller.hpp"

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
                unsigned int pixel = j*width + i;

                // initialize RNG
                // seeding strategy: the key consists of the pixel number and a unique id for this usage
                // different rngs used in different places in fresnel need different uids
                // the counter entries consist of: a counter, the ray depth, the sample index, and a user seed
                // the last 3 provide unique RNG streams for different rays and samples. The first allows
                // multiple random numbers to be produced by incrementing the counter
                r123::Philox4x32 rng;
                r123::Philox4x32::ukey_type rng_uk={{pixel, 0x11ffabcd}};
                r123::Philox4x32::key_type rng_key = rng_uk;

                // determine the viewing plane relative coordinates
                float ys = -1.0f*(j/float(height-1)-0.5f);
                float xs = i/float(height-1)-0.5f*float(width)/float(height);

                vec3<float> origin = cam.origin(xs, ys);
                vec3<float> direction = cam.direction(xs, ys);

                // determine the output pixel color
                RGB<float> c(0,0,0);
                RGB<float> attenuation(1.0f,1.0f,1.0f);
                float a = 1.0f;
                int lastprimID = -1;

                // trace a path into the scene
                for (unsigned int depth = 0; depth <= 1; depth++)
                    {
                    RTCRay ray(origin,  direction);
                    ray.tnear = 1e-3f;
                    rtcIntersect(scene->getRTCScene(), ray);

                    if (depth == 0 && !ray.hit())
                        {
                        // if the ray from the camera misses geometry, then apply the background color and alpha
                        c = background_color;
                        a = background_alpha;
                        break;
                        }

                    if (ray.hit())
                        {
                        if (depth==0)
                            lastprimID = ray.primID;
                        else
                            {
                            if (lastprimID == ray.primID)
                                {
                                std::cout << "Warning: Hit source primitive!" << std::endl;
                                }
                            lastprimID = ray.primID;
                            }

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
                            // testing: treat solid colors as emitters
                            // when the ray hits an emitter, terminate the path and evaluate the final color
                            c += attenuation * m.getColor(ray.shading_color);
                            }
                        else
                            {
                            // when the ray hits an object with a normal material, update the attenuation using the BRDF

                            // choose a random direction l to continue the path.
                            // use gaussian RNGs and sphere point picking: http://mathworld.wolfram.com/SpherePointPicking.html
                            r123::Philox4x32::ctr_type rng_counter = {{0, depth, m_n_samples, 0}};
                            r123::Philox4x32::ctr_type rng_u = rng(rng_counter, rng_key);
                            r123::float2 rng_gauss1 = r123::boxmuller(rng_u[0], rng_u[1]);
                            r123::float2 rng_gauss2 = r123::boxmuller(rng_u[2], rng_u[3]);

                            vec3<float> l(rng_gauss1.x, rng_gauss1.y, rng_gauss2.x);

                            // TODO: normalize method in VectorMath
                            l = l / std::sqrt(dot(l, l));

                            float ndotl = dot(n, l);
                            // l is generated on the whole sphere, if it points down into the surface, make it point up
                            if (ndotl < 0.0f)
                                {
                                l = -l;
                                ndotl = -ndotl;
                                }

                            attenuation *= m.brdf(l, v, n, ray.shading_color) * ndotl;

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
                            // TODO: this assumes the ray direction is normalized. Either we normalize here or ensure that
                            // is always the case
                            if (ldotd >= cos_half_angle)
                                {
                                // hit the light
                                // the division bin sin(light half angle) normalizes the lights so that
                                // a light of color 1 of any non-zero size results in an output of 1
                                // when that light is straight over the surface
                                // TODO: move these out into a precomputation per light in the light config
                                c += attenuation * lights.color[light_id] / sinf(half_angle);
                                }
                            } // end loop over lights

                        // the ray missed geometry, no more tracing to do
                        break;
                        }
                    } // end depth loop

                // take the current sample and compute the average with the previous samples
                // the 2pi comes from factoring out the division by the pdf of 1/2pi
                // from the uniform hemisphere sampling (the phi term)
                RGBA<float> output_sample(c*2.0f*float(M_PI), a);

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
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("getNumSamples", &TracerPath::getNumSamples)
        .def("reset", &TracerPath::reset)
        ;
    }

} } // end namespace fresnel::cpu
