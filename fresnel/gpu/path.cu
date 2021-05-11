// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "TracerIDs.h"
#include "common/Camera.h"
#include "common/Light.h"
#include "common/Material.h"
#include "common/RayGen.h"
#include "common/TracerPathMethods.h"
#include <optix.h>

using namespace fresnel;

///////////////////////////////////////////////////////////////////////////////////////////
// scene wide variables

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(PRDpath, prd_path, rtPayload, );
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
RT_PROGRAM void path_exception()
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
rtDeclareVariable(unsigned int, seed, , );
rtDeclareVariable(unsigned int, n_samples, , );
rtDeclareVariable(unsigned int, light_samples, , );

//! Trace rays for Path tracer
/*! Implement Path tracer ray generation
 */
RT_PROGRAM void path_ray_gen()
    {
    // determine the viewing plane relative coordinates
    optix::size_t2 screen = linear_output_buffer.size();

    // create the ray generator for this pixel
    RayGen ray_gen(launch_index.x, launch_index.y, screen.x, screen.y, seed);

    // per ray data
    PRDpath prd;
    prd.result = RGB<float>(0, 0, 0);
    prd.a = 1.0f;

    // trace a path from the camera into the scene light_samples times
    for (prd.light_sample = 0; prd.light_sample < light_samples; prd.light_sample++)
        {
        prd.attenuation = RGB<float>(1.0f, 1.0f, 1.0f);
        prd.done = false;
        cam.generateRay(prd.origin, prd.direction, launch_index.x, launch_index.y, n_samples);

        for (prd.depth = 0;; prd.depth++)
            {
            // (1 is for the path tracer ray id)
            optix::Ray ray(prd.origin, prd.direction, TRACER_PATH_RAY_ID, scene_epsilon);
            rtTrace(top_object, ray, prd);

            // break out of the loop when done
            if (prd.done)
                break;
            } // end depth loop
        } // end light samples loop

    RGBA<float> output_sample(prd.result / float(light_samples), prd.a);

    // running average using Welford's method. Variance is not particularly useful to users,
    // so don't compute that.
    // (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
    float4 old_mean_f = linear_output_buffer[launch_index];
    RGBA<float> old_mean = RGBA<float>(old_mean_f.x, old_mean_f.y, old_mean_f.z, old_mean_f.w);
    RGBA<float> output_pixel = old_mean + (output_sample - old_mean) / float(n_samples);

    // output pixels occasionally evaluate to NaN in the GPU code path, ignore these
    if (!isfinite(output_pixel.r) || !isfinite(output_pixel.g) || !isfinite(output_pixel.b))
        output_pixel = old_mean;

    linear_output_buffer[launch_index]
        = make_float4(output_pixel.r, output_pixel.g, output_pixel.b, output_pixel.a);

    // convert the current average output to sRGB
    RGBA<unsigned char> srgb_output_pixel(0, 0, 0, 0);
    if (!highlight_warning
        || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
        srgb_output_pixel = sRGB(output_pixel);
    else
        srgb_output_pixel = sRGB(RGBA<float>(highlight_warning_color, output_pixel.a));

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
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

//! Determine result color
/*! Implement Whitted ray material
 */
RT_PROGRAM void path_closest_hit()
    {
    optix::size_t2 screen = linear_output_buffer.size();
    RayGen ray_gen(launch_index.x, launch_index.y, screen.x, screen.y, seed);

    vec3<float> ray_origin(ray.origin);
    vec3<float> ray_direction(ray.direction);

    path_tracer_hit(prd_path,
                    material,
                    outline_material,
                    shading_distance,
                    outline_width,
                    shading_color,
                    shading_normal,
                    ray_origin,
                    ray_direction,
                    t_hit,
                    ray_gen,
                    n_samples,
                    light_samples);
    }

RT_PROGRAM void path_miss()
    {
    vec3<float> dir(ray.direction);

    path_tracer_miss(prd_path, background_color, background_alpha, light_samples, lights, dir);
    }
