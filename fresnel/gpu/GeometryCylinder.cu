// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/IntersectCylinder.h"
#include "common/VectorMath.h"
#include <optix_world.h>

using namespace optix;
using namespace fresnel;

rtBuffer<float3, 2> cylinder_points;
rtBuffer<float> cylinder_radius;
rtBuffer<float3, 2> cylinder_color;

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
    const vec3<float> A(cylinder_points[make_size_t2(0, primIdx)]);
    const vec3<float> B(cylinder_points[make_size_t2(1, primIdx)]);
    const float radius = cylinder_radius[primIdx];
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);

    float t = 0, d = 0;
    vec3<float> N;
    unsigned int color_index;
    if (!intersect_ray_spherocylinder(t,
                                      d,
                                      N,
                                      color_index,
                                      ray_origin,
                                      ray_direction,
                                      A,
                                      B,
                                      radius))
        return;

    if (rtPotentialIntersection(t))
    {
        shading_normal = N;
        shading_distance = d;
        shading_color = RGB<float>(cylinder_color[make_size_t2(color_index, primIdx)]);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const vec3<float> A(cylinder_points[make_size_t2(0, primIdx)]);
    const vec3<float> B(cylinder_points[make_size_t2(1, primIdx)]);
    const float radius = cylinder_radius[primIdx];

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (radius > 0.0f && !isinf(radius))
    {
        aabb->m_min.x = min(A.x - radius, B.x - radius);
        aabb->m_min.y = min(A.y - radius, B.y - radius);
        aabb->m_min.z = min(A.z - radius, B.z - radius);
        aabb->m_max.x = max(A.x + radius, B.x + radius);
        aabb->m_max.y = max(A.y + radius, B.y + radius);
        aabb->m_max.z = max(A.z + radius, B.z + radius);
    }
    else
    {
        aabb->invalidate();
    }
}
