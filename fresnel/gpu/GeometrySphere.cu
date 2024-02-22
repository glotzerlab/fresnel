// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/IntersectSphere.h"
#include "common/VectorMath.h"
#include <optix_world.h>

using namespace optix;
using namespace fresnel;

rtBuffer<float3> sphere_position;
rtBuffer<float> sphere_radius;
rtBuffer<float3> sphere_color;

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
    {
    const vec3<float> position(sphere_position[primIdx]);
    const float radius = sphere_radius[primIdx];
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);

    float t = 0, d = 0;
    vec3<float> N;
    if (!intersect_ray_sphere(t, d, N, ray_origin, ray_direction, position, radius))
        return;

    if (rtPotentialIntersection(t))
        {
        shading_normal = N;
        shading_distance = d;
        shading_color = RGB<float>(sphere_color[primIdx]);
        rtReportIntersection(0);
        }
    }

RT_PROGRAM void bounds(int primIdx, float result[6])
    {
    const float3 cen = sphere_position[primIdx];
    const float rad = sphere_radius[primIdx];

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (rad > 0.0f && !isinf(rad))
        {
        aabb->m_min = cen - rad;
        aabb->m_max = cen + rad;
        }
    else
        {
        aabb->invalidate();
        }
    }
