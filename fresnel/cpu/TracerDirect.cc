// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerDirect.h"
#include "common/RayGen.h"
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
    arena->execute([&]{ rtcCommitScene(scene->getRTCScene()); });
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
                // create the ray generator for this pixel
                RayGen ray_gen(i, j, width, height, m_seed);

                // loop over AA samples
                RGBA<float> output_avg(0,0,0,0);
                float aa_factor_total = 0.0f;

                for (unsigned int si=0; si < m_aa_n; si++) for (unsigned int sj=0; sj < m_aa_n; sj++)
                    {
                    // determine the sample location
                    float aa_factor = 1.0f;
                    vec2<float> sample_loc = ray_gen.jitterSampleAA(aa_factor, si, sj, m_aa_n);

                    // trace a ray into the scene
                    RTCRayHit ray_hit;
                    RTCRay &ray = ray_hit.ray;
                    const vec3<float>& org = cam.origin(sample_loc);
                    ray.org_x = org.x;
                    ray.org_y = org.y;
                    ray.org_z = org.z;

                    const vec3<float>& dir = cam.direction(sample_loc);
                    ray.dir_x = dir.x;
                    ray.dir_y = dir.y;
                    ray.dir_z = dir.z;

                    ray.tnear = 0.0f;
                    ray.tfar = std::numeric_limits<float>::infinity();
                    ray.time = 0.0f;
                    ray.flags = 0;
                    ray.mask = -1;
                    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                    ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                    FresnelRTCIntersectContext context;
                    rtcInitIntersectContext(&context.context);

                    rtcIntersect1(scene->getRTCScene(), &context.context, &ray_hit);

                    // determine the output pixel color
                    RGB<float> c = background_color;
                    float a = background_alpha;

                    if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                        {
                        vec3<float> n(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z);
                        n /= std::sqrt(dot(n,n));
                        vec3<float> v = -dir / std::sqrt(dot(dir, dir));
                        Material m;

                        // apply the material color or outline color depending on the distance to the edge
                        if (context.d > scene->getOutlineWidth(ray_hit.hit.geomID))
                            m = scene->getMaterial(ray_hit.hit.geomID);
                        else
                            m = scene->getOutlineMaterial(ray_hit.hit.geomID);

                        if (m.isSolid())
                            {
                            c = m.getColor(context.shading_color);
                            }
                        else
                            {
                            c = RGB<float>(0,0,0);
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
                                    f_d = m.brdf_diffuse(l, v, n, context.shading_color) * ndotl;
                                else
                                    f_d = RGB<float>(0.0f,0.0f,0.0f);

                                RGB<float> f_s;
                                if (dot(n, r) >= 0.0f)
                                    {
                                    f_s = m.brdf_specular(r, v, n, context.shading_color, half_angle) * dot(n, r);
                                    }
                                else
                                    f_s = RGB<float>(0.0f,0.0f,0.0f);

                                c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
                                }
                            }

                        a = 1.0;
                        }

                    // accumulate filtered average
                    output_avg += RGBA<float>(c, a) * aa_factor;
                    aa_factor_total += aa_factor;
                    } // end loop over AA samples

                // write the output pixel
                unsigned int pixel = j*width + i;
                RGBA<float> output_pixel = output_avg / aa_factor_total;

                linear_output[pixel] = output_pixel;
                if (!m_highlight_warning || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
                    srgb_output[pixel] = sRGB(output_pixel);
                else
                    srgb_output[pixel] = sRGB(RGBA<float>(m_highlight_warning_color, output_pixel.a));
                } // end loop over pixels in the tile
            } // loop over tiles in this region
        }); // end parallel loop over tiles
    }); // end parallel arena

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, std::shared_ptr<TracerDirect> >(m, "TracerDirect", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("setAntialiasingN", &TracerDirect::setAntialiasingN)
        .def("getAntialiasingN", &TracerDirect::getAntialiasingN)
        ;
    }

} } // end namespace fresnel::cpu
