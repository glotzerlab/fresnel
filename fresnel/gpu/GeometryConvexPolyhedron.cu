// Copyright (c) 2016-2021 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/VectorMath.h"
#include <optix_world.h>

#include <float.h>

using namespace optix;
using namespace fresnel;

// convex polyhedron geometry buffers and variables
rtBuffer<float3> convex_polyhedron_plane_origin;
rtBuffer<float3> convex_polyhedron_plane_normal;
rtBuffer<float3> convex_polyhedron_plane_color;
rtBuffer<float3> convex_polyhedron_position;
rtBuffer<float4> convex_polyhedron_orientation;
rtBuffer<float3> convex_polyhedron_color;

rtDeclareVariable(float, convex_polyhedron_radius, , );
rtDeclareVariable(float, convex_polyhedron_color_by_face, , );

// attributes to pass on to hit programs
rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

static __device__ float getShadingDistance(vec3<float> n_hit,
                                           vec3<float> p_hit,
                                           vec3<float> ray_dir_local,
                                           vec3<float> r_hit,
                                           int n_planes)
    {
    float min_d = FLT_MAX;

    // edges come from intersections of planes
    // loop over all planes and find the intersection with the hit plane
    for (int i = 0; i < n_planes; ++i)
        {
        vec3<float> n = vec3<float>(convex_polyhedron_plane_normal[i]);
        vec3<float> p = vec3<float>(convex_polyhedron_plane_origin[i]);

        // ********
        // find the line of intersection between the two planes
        // adapted from: http://geomalgorithms.com/a05-_intersect-1.html

        // direction of the line
        vec3<float> u = cross(n, n_hit);

        // if the planes are not coplanar
        if (fabs(dot(u, u)) >= 1e-5)
            {
            int maxc; // max coordinate
            if (fabs(u.x) > fabs(u.y))
                {
                if (fabs(u.x) > fabs(u.z))
                    maxc = 1;
                else
                    maxc = 3;
                }
            else
                {
                if (fabs(u.y) > fabs(u.z))
                    maxc = 2;
                else
                    maxc = 3;
                }

            // a point on the line
            vec3<float> x0;
            float d1 = -dot(n, p);
            float d2 = -dot(n_hit, p_hit);

            // solve the problem in different ways based on which direction is maximum
            switch (maxc)
                {
            case 1: // intersect with x=0
                x0.x = 0;
                x0.y = (d2 * n.z - d1 * n_hit.z) / u.x;
                x0.z = (d1 * n_hit.y - d2 * n.y) / u.x;
                break;
            case 2: // intersect with y=0
                x0.x = (d1 * n_hit.z - d2 * n.z) / u.y;
                x0.y = 0;
                x0.z = (d2 * n.x - d1 * n_hit.x) / u.y;
                break;
            case 3: // intersect with z=0
                x0.x = (d2 * n.y - d1 * n_hit.y) / u.z;
                x0.y = (d1 * n_hit.x - d2 * n.x) / u.z;
                x0.z = 0;
                }

            // we want the distance in the view plane for consistent line edge widths
            // project the line x0 + t*u into the plane perpendicular to the view direction passing
            // through r_hit
            vec3<float> view = -ray_dir_local / sqrtf(dot(ray_dir_local, ray_dir_local));
            u = u - dot(u, view) * view;
            vec3<float> w = x0 - r_hit;
            vec3<float> w_perp = w - dot(w, view) * view;
            x0 = r_hit + w_perp;

            // ********
            // find the distance from the hit point to the line
            // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
            vec3<float> v = cross(u, x0 - r_hit);
            float dsq = dot(v, v) / dot(u, u);
            float d = sqrtf(dsq);
            if (d < min_d)
                min_d = d;
            }
        }

    return min_d;
    }

RT_PROGRAM void intersect(int item)
    {
    // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
    int n_planes = convex_polyhedron_plane_origin.size();
    float t0 = -FLT_MAX;
    float t1 = FLT_MAX;

    const vec3<float> pos_world = vec3<float>(convex_polyhedron_position[item]);
    const quat<float> q_world = quat<float>(convex_polyhedron_orientation[item]);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.direction));
    vec3<float> ray_org_local = rotate(conj(q_world), vec3<float>(ray.origin) - pos_world);

    vec3<float> t0_n_local(0, 0, 0), t0_p_local(0, 0, 0);
    vec3<float> t1_n_local(0, 0, 0), t1_p_local(0, 0, 0);
    int t0_plane_hit = 0, t1_plane_hit = 0;
    for (int i = 0; i < n_planes && t0 < t1; ++i)
        {
        vec3<float> n = vec3<float>(convex_polyhedron_plane_normal[i]);
        vec3<float> p = vec3<float>(convex_polyhedron_plane_origin[i]);

        float d = -dot(n, p);
        float denom = dot(n, ray_dir_local);
        float t = -(d + dot(n, ray_org_local)) / denom;

        // if the ray is parallel to the plane, there is no intersection when the ray is outside the
        // shape
        if (fabs(denom) < 1e-5)
            {
            if (dot(ray_org_local - p, n) > 0)
                return;
            }
        else if (denom < 0)
            {
            // find the last plane this ray enters
            if (t > t0)
                {
                t0 = t;
                t0_n_local = n;
                t0_p_local = p;
                t0_plane_hit = i;
                }
            }
        else
            {
            // find the first plane this ray exits
            if (t < t1)
                {
                t1 = t;
                t1_n_local = n;
                t1_p_local = p;
                t1_plane_hit = i;
                }
            }
        }

    // if the ray enters after it exits, it missed the polyhedron
    if (t0 > t1)
        return;

    // if the t0 is a potential intersection, we hit the entry plane
    if (rtPotentialIntersection(t0))
        {
        vec3<float> n_hit, p_hit;
        shading_normal = rotate(q_world, t0_n_local);
        n_hit = t0_n_local;
        p_hit = t0_p_local;
        shading_color = lerp(convex_polyhedron_color_by_face,
                             RGB<float>(convex_polyhedron_color[item]),
                             RGB<float>(convex_polyhedron_plane_color[t0_plane_hit]));
        shading_distance = getShadingDistance(n_hit,
                                              p_hit,
                                              ray_dir_local,
                                              ray_org_local + t0 * ray_dir_local,
                                              n_planes);
        rtReportIntersection(0);
        }
    else if (rtPotentialIntersection(t1))
        {
        vec3<float> n_hit, p_hit;
        // if t1 is a potential intersection, we hit the exit plane
        shading_normal = rotate(q_world, t1_n_local);
        n_hit = t1_n_local;
        p_hit = t1_p_local;
        shading_color = lerp(convex_polyhedron_color_by_face,
                             RGB<float>(convex_polyhedron_color[item]),
                             RGB<float>(convex_polyhedron_plane_color[t1_plane_hit]));
        shading_distance = 10.0f;
        rtReportIntersection(0);
        }
    }

RT_PROGRAM void bounds(int item, float result[6])
    {
    optix::Aabb* aabb = (optix::Aabb*)result;

    vec3<float> p = vec3<float>(convex_polyhedron_position[item]);

    aabb->m_min.x = p.x - convex_polyhedron_radius;
    aabb->m_min.y = p.y - convex_polyhedron_radius;
    aabb->m_min.z = p.z - convex_polyhedron_radius;

    aabb->m_max.x = p.x + convex_polyhedron_radius;
    aabb->m_max.y = p.y + convex_polyhedron_radius;
    aabb->m_max.z = p.z + convex_polyhedron_radius;

    if (convex_polyhedron_radius <= 0.0f)
        aabb->invalidate();
    }
