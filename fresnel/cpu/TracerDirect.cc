// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerDirect.h"
#include "common/RayGen.h"
#include <cmath>
#include <stdexcept>

#include "tbb/parallel_for.h"

using namespace tbb;

namespace fresnel
    {
namespace cpu
    {
/*! \param device Device to attach the raytracer to
 */
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    }

TracerDirect::~TracerDirect() { }

void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    std::shared_ptr<tbb::task_arena> arena = scene->getDevice()->getTBBArena();

    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    // direct tracers do not support depth of field
    UserCamera user_camera = scene->getCamera();
    user_camera.f_stop = std::numeric_limits<float>::infinity();

    // disable aa sampling with m_aa_n is 1
    bool sample_aa = (m_aa_n != 1);

    const Camera cam(user_camera, m_linear_out->getW(), m_linear_out->getH(), m_seed, sample_aa);
    const Lights lights(scene->getLights(), cam);
    Tracer::render(scene);

    // update Embree data structures
    rtcCommitScene(scene->getRTCScene());
    m_device->checkError();

    RGBA<float>* linear_output = m_linear_out->map();
    RGBA<unsigned char>* srgb_output = m_srgb_out->map();

    // for each pixel
    const unsigned int height = m_linear_out->getH();
    const unsigned int width = m_linear_out->getW();

    const unsigned int TILE_SIZE_X = 4;
    const unsigned int TILE_SIZE_Y = 4;
    const unsigned int numTilesX = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const unsigned int numTilesY = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    arena->execute([&] {
        parallel_for(
            blocked_range<size_t>(0, numTilesX * numTilesY),
            [=](const blocked_range<size_t>& r) {
                for (size_t tile = r.begin(); tile != r.end(); ++tile)
                    {
                    const unsigned int tileY = tile / numTilesX;
                    const unsigned int tileX = tile - tileY * numTilesX;
                    const unsigned int x0 = tileX * TILE_SIZE_X;
                    const unsigned int x1 = std::min(x0 + TILE_SIZE_X, width);
                    const unsigned int y0 = tileY * TILE_SIZE_Y;
                    const unsigned int y1 = std::min(y0 + TILE_SIZE_Y, height);

                    for (unsigned int j = y0; j < y1; j++)
                        for (unsigned int i = x0; i < x1; i++)
                            {
                            // create the ray generator for this pixel
                            RayGen ray_gen(i, j, width, height, m_seed);

                            // loop over AA samples
                            RGBA<float> output_avg(0, 0, 0, 0);

                            for (unsigned int sample = 0; sample < m_aa_n * m_aa_n; sample++)
                                {
                                // trace a ray into the scene
                                RTCRayHit ray_hit;
                                RTCRay& ray = ray_hit.ray;
                                vec3<float> org, dir;
                                cam.generateRay(org, dir, i, j, sample);
                                ray.org_x = org.x;
                                ray.org_y = org.y;
                                ray.org_z = org.z;

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
                                    vec3<float> n(ray_hit.hit.Ng_x,
                                                  ray_hit.hit.Ng_y,
                                                  ray_hit.hit.Ng_z);
                                    n /= std::sqrt(dot(n, n));
                                    vec3<float> v = -dir / std::sqrt(dot(dir, dir));
                                    Material m;

                                    // apply the material color or outline color depending on
                                    // the distance to the edge
                                    if (context.d >= scene->getOutlineWidth(ray_hit.hit.geomID))
                                        m = scene->getMaterial(ray_hit.hit.geomID);
                                    else
                                        m = scene->getOutlineMaterial(ray_hit.hit.geomID);

                                    if (m.isSolid())
                                        {
                                        c = m.getColor(context.shading_color);
                                        }
                                    else
                                        {
                                        c = RGB<float>(0, 0, 0);
                                        for (unsigned int light_id = 0; light_id < lights.N;
                                             light_id++)
                                            {
                                            vec3<float> l = lights.direction[light_id];

                                            // find the representative point, a vector pointing
                                            // to the a point on the area light with a smallest
                                            // angle to the reflection vector
                                            vec3<float> r = -v + (2.0f * n * dot(n, v));

                                            // find the closest point on the area light
                                            float half_angle = lights.theta[light_id];
                                            float cos_half_angle = cosf(half_angle);
                                            float ldotr = dot(l, r);
                                            if (ldotr < cos_half_angle)
                                                {
                                                vec3<float> a = cross(l, r);
                                                a = a / sqrtf(dot(a, a));

                                                // miss the light, need to rotate r by the
                                                // difference in the angles about l cross r
                                                quat<float> q = quat<float>::fromAxisAngle(
                                                    a,
                                                    -acosf(ldotr) + half_angle);
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
                                                f_d = m.brdf_diffuse(l, v, n, context.shading_color)
                                                      * ndotl;
                                            else
                                                f_d = RGB<float>(0.0f, 0.0f, 0.0f);

                                            RGB<float> f_s;
                                            if (dot(n, r) >= 0.0f)
                                                {
                                                f_s = m.brdf_specular(r,
                                                                      v,
                                                                      n,
                                                                      context.shading_color,
                                                                      half_angle)
                                                      * dot(n, r);
                                                }
                                            else
                                                f_s = RGB<float>(0.0f, 0.0f, 0.0f);

                                            c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
                                            }
                                        }

                                    a = 1.0;
                                    }

                                // accumulate importance sampled average
                                output_avg += RGBA<float>(c, a);
                                } // end loop over AA samples

                            // write the output pixel
                            unsigned int pixel = j * width + i;
                            RGBA<float> output_pixel = output_avg / float(m_aa_n * m_aa_n);

                            linear_output[pixel] = output_pixel;
                            if (!m_highlight_warning
                                || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f
                                    && output_pixel.b <= 1.0f))
                                srgb_output[pixel] = sRGB(output_pixel);
                            else
                                srgb_output[pixel]
                                    = sRGB(RGBA<float>(m_highlight_warning_color, output_pixel.a));
                            } // end loop over pixels in the tile
                    }         // loop over tiles in this region
            });           // end parallel loop over tiles
    });                   // end parallel arena

    m_linear_out->unmap();
    m_srgb_out->unmap();
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, Tracer, std::shared_ptr<TracerDirect>>(m, "TracerDirect")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("setAntialiasingN", &TracerDirect::setAntialiasingN)
        .def("getAntialiasingN", &TracerDirect::getAntialiasingN);
    }

    } // namespace cpu
    } // namespace fresnel
