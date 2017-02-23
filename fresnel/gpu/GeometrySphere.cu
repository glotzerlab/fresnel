// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix_world.h>
#include "common/VectorMath.h"

using namespace optix;
using namespace fresnel;

rtBuffer<float3> sphere_position;
rtBuffer<float> sphere_radius;
rtBuffer<float3> sphere_color;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(float3, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
    {
    const vec3<float> position(sphere_position[primIdx]);
    const float radius = sphere_radius[primIdx];
    const float rsq = (radius)*(radius);
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);
    const vec3<float> v = ray_origin-position;
    const vec3<float> w = cross(ray_direction,v);

    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(ray_direction,ray_direction);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(dot(v,v) - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    const float t0 = Rp-Ri;
    const float t1 = Rp+Ri;

    if (rtPotentialIntersection(t0))
        {
        shading_normal = ray_origin+t0*ray_direction-position;
        shading_distance = radius - sqrt(Dsq);
        shading_color = sphere_color[primIdx];
        rtReportIntersection(0);
        }
    else if (rtPotentialIntersection(t1))
        {
        shading_normal = ray_origin+t1*ray_direction-position;
        shading_color = sphere_color[primIdx];
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

