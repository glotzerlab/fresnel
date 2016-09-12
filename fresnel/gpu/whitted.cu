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
    float a = 0.0;
    if (prd.hit)
        {
        c = RGB<float>(prd.result);
        a = 1.0;
        }

    // write the output pixel
    output_buffer[launch_index] = make_float4(c.r, c.g, c.b, a);
    }

///////////////////////////////////////////////////////////////////////////////////////////
// closest hit variables

rtDeclareVariable(float3, material_color, , );
rtDeclareVariable(float, material_solid, , );

//! Determine result color
/*! Implement Whitted ray material
*/
RT_PROGRAM void whitted_closest_hit()
    {
    Material m;
    m.solid = material_solid;
    m.color = RGB<float>(material_color);

    prd_radiance.result = m.luminance();
    prd_radiance.hit = 1;
    }
