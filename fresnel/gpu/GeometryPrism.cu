// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <optix_world.h>
#include "common/VectorMath.h"

#include <float.h>

using namespace optix;

// prism geometry buffers and variables
rtBuffer<float3> prism_plane_origin;
rtBuffer<float3> prism_plane_normal;
rtBuffer<float2> prism_position;
rtBuffer<float> prism_angle;
rtBuffer<float> prism_height;
rtBuffer<float3> prism_color;

rtDeclareVariable(float, prism_radius, , );

// attributes to pass on to hit programs
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(float3, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int item)
    {
    // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
    int n_planes = prism_plane_origin.size();
    float t0 = -FLT_MAX;
    float t1 = FLT_MAX;

    const vec2<float> p2 = vec2<float>(prism_position[item]);
    const vec3<float> pos_world(p2.x, p2.y, 0.0f);
    const float angle = prism_angle[item];
    const float height = prism_height[item];
    const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0,0,1), angle);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.direction));
    vec3<float> ray_org_local = rotate(conj(q_world), vec3<float>(ray.origin) - pos_world);

    vec3<float> t0_n_local, t0_p_local;
    vec3<float> t1_n_local, t1_p_local;
    for(int i = 0; i < n_planes && t0 < t1; ++i )
        {
        vec3<float> n = vec3<float>(prism_plane_normal[i]);
        vec3<float> p = vec3<float>(prism_plane_origin[i]);

        // correct the top plane positions
        if (i == 0)
            p.z = height;

        float d = -dot(n, p);
        float denom = dot(n, ray_dir_local);
        float t = -(d + dot(n, ray_org_local))/denom;

        // if the ray is parallel to the plane, there is no intersection when the ray is outside the shape
        if (fabs(denom) < 1e-5)
            {
            if (dot(ray_org_local - p, n) > 0)
                return;
            }
        else if (denom < 0)
            {
            // find the last plane this ray enters
            if(t > t0)
                {
                t0 = t;
                t0_n_local = n;
                t0_p_local = p;
                }
            }
        else
            {
            // find the first plane this ray exits
            if(t < t1)
                {
                t1 = t;
                t1_n_local = n;
                t1_p_local = p;
                }
            }
        }

    // if the ray enters after it exits, it missed the polyhedron
    if(t0 > t1)
        return;

    // otherwise, it hit: fill out the hit structure and track the plane that was hit
    float t_hit = 0;
    bool hit = false;
    vec3<float> n_hit, p_hit;

    // if the t0 is a potential intersection, we hit the entry plane
    if (rtPotentialIntersection(t0))
        {
        shading_normal = rotate(q_world, t0_n_local);
        n_hit = t0_n_local;
        p_hit = t0_p_local;
        shading_color = prism_color[item];
        hit = true;
        shading_distance = 10.0f;
        rtReportIntersection(0);
        }
    else if (rtPotentialIntersection(t1))
        {
        // if t1 is a potential intersection, we hit the exit plane
        shading_normal = rotate(q_world, t1_n_local);
        n_hit = t1_n_local;
        p_hit = t1_p_local;
        shading_color = prism_color[item];
        hit = true;
        shading_distance = 10.0f;
        rtReportIntersection(0);
        }

    // TODO: shading_distance
    }


RT_PROGRAM void bounds (int item, float result[6])
    {
    optix::Aabb* aabb = (optix::Aabb*)result;

    vec2<float> p2 = vec2<float>(prism_position[item]);
    float height = prism_height[item];
    vec3<float> p(p2.x, p2.y, 0.0f);

    aabb->m_min.x = p.x - prism_radius;
    aabb->m_min.y = p.y - prism_radius;
    aabb->m_min.z = p.z;

    aabb->m_max.x = p.x + prism_radius;
    aabb->m_max.y = p.y + prism_radius;
    aabb->m_max.z = p.z + height;

    if (height <= 0.0f)
        aabb->invalidate();
    }

