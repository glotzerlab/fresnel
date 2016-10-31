// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix.h>
#include "common/Material.h"
#include "common/Camera.h"

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

rtBuffer<float4, 2> output_buffer;

///////////////////////////////////////////////////////////////////////////////////////////
// variables output from intersection program

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(float3, shading_color, attribute shading_color, );


///////////////////////////////////////////////////////////////////////////////////////////

//! Report exceptions
/*! Stack overflows result in magenta pixels, other exceptions are printed.
*/
RT_PROGRAM void whitted_exception()
    {
    const unsigned int code = rtGetExceptionCode();

    if(code == RT_EXCEPTION_STACK_OVERFLOW)
        output_buffer[launch_index] = make_float4(bad_color.x, bad_color.y, bad_color.z, 1.0f);
    else
        rtPrintExceptionDetails();
    }

///////////////////////////////////////////////////////////////////////////////////////////
// ray gen variables

rtDeclareVariable(float3, camera_p, , );
rtDeclareVariable(float3, camera_d, , );
rtDeclareVariable(float3, camera_u, , );
rtDeclareVariable(float3, camera_r, , );
rtDeclareVariable(float, camera_h, , );

//! Trace rays for Whitted
/*! Implement Whitted ray generation
*/
RT_PROGRAM void whitted_ray_gen()
    {
    // load camera
    Camera cam;
    cam.p = vec3<float>(camera_p);
    cam.d = vec3<float>(camera_d);
    cam.u = vec3<float>(camera_u);
    cam.r = vec3<float>(camera_r);
    cam.h = camera_h;

    // determine the viewing plane relative coordinates
    optix::size_t2 screen = output_buffer.size();
    float ys = -1.0f*(launch_index.y/float(screen.y-1)-0.5f);
    float xs = launch_index.x/float(screen.y-1)-0.5f*float(screen.x)/float(screen.y);

    // trace a ray into the scene
    optix::Ray ray(cam.origin(xs, ys), cam.direction(xs, ys), 0, scene_epsilon);

    PRDradiance prd;
    prd.hit = 0;

    rtTrace(top_object, ray, prd);

    // determine the output pixel color
    RGB<float> c;
    float a = 0.0f;
    if (prd.hit)
        {
        c = RGB<float>(prd.result);
        a = 1.0f;
        }

    // write the output pixel
    output_buffer[launch_index] = make_float4(c.r, c.g, c.b, a);
    }

///////////////////////////////////////////////////////////////////////////////////////////
// closest hit variables

rtDeclareVariable(float3, material_color, , );
rtDeclareVariable(float, material_solid, , );
rtDeclareVariable(float, material_geometry_color_mix, , );

//! Determine result color
/*! Implement Whitted ray material
*/
RT_PROGRAM void whitted_closest_hit()
    {
    Material m;

    // apply the material color or outline color depending on the distance to the edge
    if (shading_distance > 0.05)
        {
        m.solid = material_solid;
        m.color = RGB<float>(material_color);
        m.geometry_color_mix = material_geometry_color_mix;
        }
    else
        {
        m.solid = 1.0;
        m.color = RGB<float>(0,0,0);
        m.geometry_color_mix = 0;
        }

    vec3<float> Ng(shading_normal);
    vec3<float> n = Ng * rsqrtf(dot(Ng, Ng));
    vec3<float> l = vec3<float>(1,1,1);
    l = l * rsqrtf(dot(l,l));
    vec3<float> dir = vec3<float>(ray.direction);
    vec3<float> v = -dir * rsqrtf(dot(dir, dir));

    RGB<float> c;

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
        }

    prd_radiance.result = c;
    prd_radiance.hit = 1;
    }
