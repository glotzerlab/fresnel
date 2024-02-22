// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "TracerIDs.h"
#include "common/Camera.h"
#include "common/Light.h"
#include "common/Material.h"
#include "common/RayGen.h"
#include <optix.h>

using namespace fresnel;

///////////////////////////////////////////////////////////////////////////////////////////
// scene wide variables

//! Per ray data for radiance rays
struct alignas(16) PRDradiance
    {
    float3 result;
    unsigned int hit;
    };

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(PRDradiance, prd_radiance, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtBuffer<float4, 2> linear_output_buffer;
rtBuffer<uchar4, 2> srgb_output_buffer;

///////////////////////////////////////////////////////////////////////////////////////////
// variables output from intersection program

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );

///////////////////////////////////////////////////////////////////////////////////////////

//! Report exceptions
/*! Stack overflows result in magenta pixels, other exceptions are printed.
 */
RT_PROGRAM void direct_exception()
    {
    const unsigned int code = rtGetExceptionCode();

    if (code == RT_EXCEPTION_STACK_OVERFLOW)
        {
        linear_output_buffer[launch_index]
            = make_float4(bad_color.x, bad_color.y, bad_color.z, 1.0f);
        srgb_output_buffer[launch_index]
            = make_uchar4(255.0f * bad_color.x, 255.0f * bad_color.y, 255.0f * bad_color.z, 255);
        }
    else
        rtPrintExceptionDetails();
    }

///////////////////////////////////////////////////////////////////////////////////////////
// ray gen variables

rtDeclareVariable(Camera, cam, , );
rtDeclareVariable(RGB<float>, background_color, , );
rtDeclareVariable(RGB<float>, highlight_warning_color, , );
rtDeclareVariable(unsigned int, highlight_warning, , );
rtDeclareVariable(float, background_alpha, , );
rtDeclareVariable(Lights, lights, , );
rtDeclareVariable(unsigned int, aa_n, , );
rtDeclareVariable(unsigned int, seed, , );

//! Trace rays for Whitted
/*! Implement Whitted ray generation
 */
RT_PROGRAM void direct_ray_gen()
    {
    // determine the viewing plane relative coordinates
    optix::size_t2 screen = linear_output_buffer.size();

    // create the ray generator for this pixel
    RayGen ray_gen(launch_index.x, launch_index.y, screen.x, screen.y, seed);

    // loop over AA samples
    RGBA<float> output_avg(0, 0, 0, 0);

    for (unsigned int sample = 0; sample < aa_n * aa_n; sample++)
        {
        // trace a ray into the scene
        vec3<float> org, dir;
        cam.generateRay(org, dir, launch_index.x, launch_index.y, sample);

        optix::Ray ray(org, dir, TRACER_PREVIEW_RAY_ID, scene_epsilon);

        PRDradiance prd;
        prd.hit = 0;

        rtTrace(top_object, ray, prd);

        // determine the output pixel color
        RGB<float> c(background_color);
        float a = background_alpha;
        if (prd.hit)
            {
            c = RGB<float>(prd.result);
            a = 1.0f;
            }

        // accumulate importance sampled average
        output_avg += RGBA<float>(c, a);
        }

    // correct aa sample average
    RGBA<float> output_pixel = output_avg / float(aa_n * aa_n);

    // write the output pixel
    RGBA<unsigned char> srgb_output_pixel(0, 0, 0, 0);
    if (!highlight_warning
        || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
        srgb_output_pixel = sRGB(output_pixel);
    else
        srgb_output_pixel = sRGB(RGBA<float>(highlight_warning_color, output_pixel.a));

    linear_output_buffer[launch_index]
        = make_float4(output_pixel.r, output_pixel.g, output_pixel.b, output_pixel.a);
    srgb_output_buffer[launch_index] = make_uchar4(srgb_output_pixel.r,
                                                   srgb_output_pixel.g,
                                                   srgb_output_pixel.b,
                                                   srgb_output_pixel.a);
    }

///////////////////////////////////////////////////////////////////////////////////////////
// closest hit variables

rtDeclareVariable(Material, material, , );
rtDeclareVariable(Material, outline_material, , );
rtDeclareVariable(float, outline_width, , );

//! Determine result color
/*! Implement Whitted ray material
 */
RT_PROGRAM void direct_closest_hit()
    {
    Material m;

    // apply the material color or outline color depending on the distance to the edge
    if (shading_distance >= outline_width)
        {
        m = material;
        }
    else
        {
        m = outline_material;
        }

    vec3<float> n = shading_normal * rsqrtf(dot(shading_normal, shading_normal));
    vec3<float> dir = vec3<float>(ray.direction);
    vec3<float> v = -dir * rsqrtf(dot(dir, dir));

    RGB<float> c(0, 0, 0);

    if (m.isSolid())
        {
        c = m.getColor(shading_color);
        }
    else
        {
        c = RGB<float>(0, 0, 0);
        for (unsigned int light_id = 0; light_id < lights.N; light_id++)
            {
            vec3<float> l = lights.direction[light_id];

            // find the representative point, a vector pointing to the a point on the area light
            // with a smallest angle to the reflection vector
            vec3<float> r = -v + (2.0f * n * dot(n, v));

            // find the closest point on the area light
            float half_angle = lights.theta[light_id];
            float cos_half_angle = cosf(half_angle);
            float ldotr = dot(l, r);
            if (ldotr < cos_half_angle)
                {
                vec3<float> a = cross(l, r);
                a = a / sqrtf(dot(a, a));

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
                f_d = m.brdf_diffuse(l, v, n, shading_color) * ndotl;
            else
                f_d = RGB<float>(0.0f, 0.0f, 0.0f);

            RGB<float> f_s;
            if (dot(n, r) >= 0.0f)
                {
                f_s = m.brdf_specular(r, v, n, shading_color, half_angle) * dot(n, r);
                }
            else
                f_s = RGB<float>(0.0f, 0.0f, 0.0f);

            c += (f_d + f_s) * float(M_PI) * lights.color[light_id];
            }
        }

    prd_radiance.result = float3(c);
    prd_radiance.hit = 1;
    }
