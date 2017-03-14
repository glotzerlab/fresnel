// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix.h>
#include "common/Material.h"
#include "common/Camera.h"

using namespace fresnel;

///////////////////////////////////////////////////////////////////////////////////////////
// scene wide variables

//! Per ray data for radiance rays
struct PRDradiance
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

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(float3, shading_color, attribute shading_color, );


///////////////////////////////////////////////////////////////////////////////////////////

//! Report exceptions
/*! Stack overflows result in magenta pixels, other exceptions are printed.
*/
RT_PROGRAM void direct_exception()
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
rtDeclareVariable(float3, background_color, , );
rtDeclareVariable(float, background_alpha, , );
rtDeclareVariable(float3, light_direction, , );

//! Trace rays for Whitted
/*! Implement Whitted ray generation
*/
RT_PROGRAM void direct_ray_gen()
    {
    // determine the viewing plane relative coordinates
    optix::size_t2 screen = linear_output_buffer.size();
    float ys = -1.0f*(launch_index.y/float(screen.y-1)-0.5f);
    float xs = launch_index.x/float(screen.y-1)-0.5f*float(screen.x)/float(screen.y);

    // trace a ray into the scene
    optix::Ray ray(cam.origin(xs, ys), cam.direction(xs, ys), 0, scene_epsilon);

    PRDradiance prd;
    prd.hit = 0;

    rtTrace(top_object, ray, prd);

    // determine the output pixel color
    RGB<float> c(background_color.x, background_color.y, background_color.z);
    float a = background_alpha;
    if (prd.hit)
        {
        c = RGB<float>(prd.result);
        a = 1.0f;
        }

    // write the output pixel
    linear_output_buffer[launch_index] = make_float4(c.r, c.g, c.b, a);
    RGBA<unsigned char> srgb_output_pixel = sRGB(RGBA<float>(c.r, c.g, c.b, a));
    srgb_output_buffer[launch_index] = make_uchar4(srgb_output_pixel.r, srgb_output_pixel.g, srgb_output_pixel.b, srgb_output_pixel.a);
    }

///////////////////////////////////////////////////////////////////////////////////////////
// closest hit variables

rtDeclareVariable(float3, material_color, , );
rtDeclareVariable(float, material_solid, , );
rtDeclareVariable(float, material_primitive_color_mix, , );
rtDeclareVariable(float3, outline_material_color, , );
rtDeclareVariable(float, outline_material_solid, , );
rtDeclareVariable(float, outline_material_primitive_color_mix, , );
rtDeclareVariable(float, outline_width, , );

//! Determine result color
/*! Implement Whitted ray material
*/
RT_PROGRAM void direct_closest_hit()
    {
    Material m;

    // apply the material color or outline color depending on the distance to the edge
    if (shading_distance > outline_width)
        {
        m.solid = material_solid;
        m.color = RGB<float>(material_color);
        m.primitive_color_mix = material_primitive_color_mix;
        }
    else
        {
        m.solid = outline_material_solid;
        m.color = RGB<float>(outline_material_color);
        m.primitive_color_mix = outline_material_primitive_color_mix;
        }

    vec3<float> Ng(shading_normal);
    vec3<float> n = Ng * rsqrtf(dot(Ng, Ng));
    vec3<float> l(light_direction);
    vec3<float> dir = vec3<float>(ray.direction);
    vec3<float> v = -dir * rsqrtf(dot(dir, dir));

    RGB<float> c(1,1,1);

    if (m.isSolid())
        {
        c = m.getColor(RGB<float>(shading_color));
        }
    else
        {
        // only apply brdf when the light faces the surface
        float ndotl = dot(n,l);
        if (ndotl > 0.0f)
            {
            c = m.brdf(l, v, n, RGB<float>(shading_color)) * float(M_PI) * /* light color * */ ndotl;
            }
        else
            {
            c = RGB<float>(0,0,0);
            }
        }

    prd_radiance.result = c;
    prd_radiance.hit = 1;
    }
