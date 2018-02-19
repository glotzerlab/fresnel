// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix_world.h>
#include "common/VectorMath.h"
#include "common/ColorMath.h"

using namespace optix;
using namespace fresnel;

rtBuffer<float3> sphere_position;
rtBuffer<float> sphere_radius;
rtBuffer<float3> sphere_color;

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// TODO: There is a simpler ray-sphere intersection test in the minimal line of code path tracer

RT_PROGRAM void intersect(int primIdx)
    {
    const vec3<float> position(sphere_position[primIdx]);
    const float radius = sphere_radius[primIdx];
    const float rsq = (radius)*(radius);
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);
    const vec3<float> v = position-ray_origin;
    const float vsq = dot(v,v);
    const vec3<float> w = cross(v,ray_direction);

    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(ray_direction,ray_direction);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(vsq - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    float t;
    if (dot(v, ray_direction) > 0.0f)
        {
        if (vsq > rsq)
            {
            // ray origin is outside the sphere, compute the distance back from the closest point
            t = Rp-Ri;
            }
        else
            {
            // ray origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Rp+Ri;
            }
        }
    else
        {
        // origin is behind the sphere (use tolerance to exclude origins directly on the sphere)
        if (vsq - rsq > -3e-6f*rsq)
            {
            // origin is outside the sphere, no intersection
            return;
            }
        else
            {
            // origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Ri-Rp;
            }
        }

    if (rtPotentialIntersection(t))
        {
        shading_normal = ray_origin+t*ray_direction-position;
        shading_distance = radius - sqrt(Dsq);
        shading_color = RGB<float>(sphere_color[primIdx]);
        rtReportIntersection(0);
        }
    }


RT_PROGRAM void bounds(int primIdx, float result[6])
    {
    const float3 cen = sphere_position[primIdx];
    const float rad = sphere_radius[primIdx];

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (rad > 0.0f  && !isinf(rad))
        {
        aabb->m_min = cen - rad;
        aabb->m_max = cen + rad;
        }
    else
        {
        aabb->invalidate();
        }
    }

