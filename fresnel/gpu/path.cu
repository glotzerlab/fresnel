// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix.h>
#include "common/Material.h"
#include "common/Camera.h"
#include "common/Light.h"
#include "common/RayGen.h"

using namespace fresnel;

///////////////////////////////////////////////////////////////////////////////////////////
// scene wide variables

//! Per ray data for radiance rays
struct PRDpath
    {
    RGB<float> result;
    float a;
    RGB<float> attenuation;
    vec3<float> origin;
    vec3<float> direction;
    unsigned int light_sample;
    unsigned int depth;
    bool done;
    };

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

    if(code == RT_EXCEPTION_STACK_OVERFLOW)
        {
        linear_output_buffer[launch_index] = make_float4(bad_color.x, bad_color.y, bad_color.z, 1.0f);
        srgb_output_buffer[launch_index] = make_uchar4(255.0f*bad_color.x, 255.0f*bad_color.y, 255.0f*bad_color.z, 255);
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

    // determine the viewing plane relative coordinates of this pixel
    vec2<float> sample_loc = ray_gen.importanceSampleAA(n_samples);

    // per ray data
    PRDpath prd;
    prd.result = RGB<float>(0,0,0);
    prd.a = 1.0f;

    // trace a path from the hit point into the scene light_samples times
    for (prd.light_sample = 0; prd.light_sample < light_samples; prd.light_sample++)
        {
        prd.attenuation = RGB<float>(1.0f,1.0f,1.0f);
        prd.done = false;
        prd.origin = cam.origin(sample_loc);
        prd.direction = cam.direction(sample_loc);

        for (prd.depth = 0; prd.depth <= 10000; prd.depth++)
            {
            // (1 is for the path tracer ray id)
            optix::Ray ray(prd.origin, prd.direction, 1, scene_epsilon);
            rtTrace(top_object, ray, prd);

            // break out of the loop when done
            if (prd.done)
                break;

            // break out of the loop when attenuation is small (TODO: Russian roulette)
            if (prd.attenuation.r < 1e-6 &&
                prd.attenuation.g < 1e-6 &&
                prd.attenuation.b < 1e-6)
                {
                break;
                }
            } // end depth loop
        } // end light samples loop

    RGBA<float> output_sample(prd.result / float(light_samples), prd.a);

    // running average using Welford's method. Variance is not particularly useful to users,
    // so don't compute that.
    // (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
    float4 old_mean_f = linear_output_buffer[launch_index];
    RGBA<float> old_mean = RGBA<float>(old_mean_f.x, old_mean_f.y, old_mean_f.z, old_mean_f.w);
    RGBA<float> output_pixel = old_mean + (output_sample-old_mean)/float(n_samples);
    linear_output_buffer[launch_index] = make_float4(output_pixel.r, output_pixel.g, output_pixel.b, output_pixel.a);

    // convert the current average output to sRGB
    RGBA<unsigned char> srgb_output_pixel(0,0,0,0);
    if (!highlight_warning || (output_pixel.r <= 1.0f && output_pixel.g <= 1.0f && output_pixel.b <= 1.0f))
        srgb_output_pixel = sRGB(output_pixel);
    else
        srgb_output_pixel = sRGB(RGBA<float>(highlight_warning_color, output_pixel.a));

    srgb_output_buffer[launch_index] = make_uchar4(srgb_output_pixel.r, srgb_output_pixel.g, srgb_output_pixel.b, srgb_output_pixel.a);
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

    Material m;

    // apply the material color or outline color depending on the distance to the edge
    if (shading_distance > outline_width)
        {
        m = material;
        }
    else
        {
        m = outline_material;
        }

    vec3<float> n = shading_normal * rsqrtf(dot(shading_normal, shading_normal));
    vec3<float> v = -vec3<float>(ray.direction);

    if (m.isSolid())
        {
        // testing: treat solid colors as emitters
        // when the ray hits an emitter, terminate the path and evaluate the final color
        prd_path.result += prd_path.attenuation * m.getColor(shading_color);
        prd_path.done = true;
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
                                                          prd_path.depth,
                                                          (n_samples-1)*light_samples + prd_path.light_sample,
                                                          m);
        if (transmit)
            {
            // perfect transmission
            RGB<float> trans_color = m.getColor(shading_color);
            trans_color.r = sqrtf(trans_color.r);
            trans_color.g = sqrtf(trans_color.g);
            trans_color.b = sqrtf(trans_color.b);
            prd_path.attenuation *= trans_color;
            }
        else
            {
            float ndotl = dot(n, l);

            // When n dot v is less than 0, this is coming through the back of the surface
            // skip this sample
            if (dot(n,v) > 0.0f && ndotl > 0.0f)
                {
                prd_path.attenuation *= m.brdf(l, v, n, shading_color) *
                                               ndotl *
                                               factor;
                }
            else
                {
                prd_path.attenuation = RGB<float>(0,0,0);
                prd_path.done = true;
                }
            }

        // set the origin and direction for the next ray in the path
        prd_path.origin = vec3<float>(ray.origin) + vec3<float>(ray.direction) * t_hit;
        prd_path.direction = l;
        }
    }

RT_PROGRAM void path_miss()
    {
    if (prd_path.depth == 0)
        {
        // a miss at depth 0 implies we hit the background on the primary ray
        // no need to resample the background N times
        prd_path.result = background_color * float(light_samples);
        prd_path.a = background_alpha;
        }
    else
        {
        // on subsequent rays, process the area lights
        // see if it hit any lights and compute the output color accordingly
        for (unsigned int light_id = 0; light_id < lights.N; light_id++)
            {
            vec3<float> l = lights.direction[light_id];
            float half_angle = lights.theta[light_id];
            float cos_half_angle = cosf(half_angle);
            if (half_angle > float(M_PI)/2.0f)
                half_angle = float(M_PI)/2.0f;
            float ldotd = dot(l, vec3<float>(ray.direction));

            // NB: we can do this because ray directions are always normalized
            if (ldotd >= cos_half_angle)
                {
                // hit the light
                // the division bin sin(light half angle) normalizes the lights so that
                // a light of color 1 of any non-zero size results in an output of 1
                // when that light is straight over the surface
                prd_path.result += prd_path.attenuation * lights.color[light_id] / sinf(half_angle);
                }
            } // end loop over lights
        }

    prd_path.done = true;
    }
