// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/IntersectEllipsoid.h"
#include "common/VectorMath.h"
#include <optix_world.h>

using namespace optix;
using namespace fresnel;

rtBuffer<float3> ellipsoid_position;
rtBuffer<float3> ellipsoid_radii;
rtBuffer<quat<float>> ellipsoid_orientation;
rtBuffer<float3> ellipsoid_color;

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
    {
    const vec3<float> position(ellipsoid_position[primIdx]);
    const vec3<float> radii = vec3<float>(ellipsoid_radii[primIdx]);
    const quat<float> orientation = quat<float>(ellipsoid_orientation[primIdx]);
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);

    float t = 0, d = 0;
    vec3<float> N;
    if (!intersect_ray_ellipsoid(t, d, N, ray_origin, ray_direction, position, radii, orientation))
        return;

    if (rtPotentialIntersection(t))
        {
        shading_normal = N;
        shading_distance = d;
        shading_color = RGB<float>(ellipsoid_color[primIdx]);
        rtReportIntersection(0);
        }
    }

RT_PROGRAM void bounds(int primIdx, float result[6])
    {
    const float3 cen = ellipsoid_position[primIdx];
    const float3 radii = ellipsoid_radii[primIdx];
	// use bounding sphere for now, TODO optimized method
	const float max_radius = fmax(radii.x, fmax(radii.y, radii.z));
	// TODO does float3 have x y z attributes?
    optix::Aabb* aabb = (optix::Aabb*)result;

    if (max_radius > 0.0f && !isinf(max_radius))
        {
        aabb->m_min = cen - max_radius;
        aabb->m_max = cen + max_radius;
        }
    else
        {
        aabb->invalidate();
        }
    }
