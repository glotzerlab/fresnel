// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/GeometryMath.h"
#include "common/VectorMath.h"
#include <optix_world.h>

#include <float.h>

using namespace optix;
using namespace fresnel;

// polygon geometry buffers and variables
rtBuffer<float2> polygon_vertices;
rtBuffer<float2> polygon_position;
rtBuffer<float> polygon_angle;
rtBuffer<float3> polygon_color;

rtDeclareVariable(float, polygon_radius, , );
rtDeclareVariable(float, polygon_rounding_radius, , );

// attributes to pass on to hit programs
rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

//! Test if a point is inside a polygon
/*! \param min_d  [out] minimum distance from p to the polygon edge
    \param p Point

    Vertices are read from the geometry variable polygon_vertices.

    \returns true if the point is inside the polygon

    \note \a p is *in the polygon's reference frame!*

    \ingroup overlap
*/
static __device__ bool is_inside(float& min_d, const vec2<float>& p)
{
    // code for concave test from: http://alienryderflex.com/polygon/
    unsigned int nvert = polygon_vertices.size();
    min_d = FLT_MAX;

    unsigned int i, j = nvert - 1;
    bool oddNodes = false;

    for (i = 0; i < nvert; i++)
    {
        min_d = fast::min(min_d,
                          point_line_segment_distance(p,
                                                      vec2<float>(polygon_vertices[i]),
                                                      vec2<float>(polygon_vertices[j])));

        if ((polygon_vertices[i].y < p.y && polygon_vertices[j].y >= p.y)
            || (polygon_vertices[j].y < p.y && polygon_vertices[i].y >= p.y))
        {
            if (polygon_vertices[i].x
                    + (p.y - polygon_vertices[i].y)
                          / (polygon_vertices[j].y - polygon_vertices[i].y)
                          * (polygon_vertices[j].x - polygon_vertices[i].x)
                < p.x)
            {
                oddNodes = !oddNodes;
            }
        }
        j = i;
    }

    return oddNodes;
}

RT_PROGRAM void intersect(int item)
{
    const vec2<float> p2 = vec2<float>(polygon_position[item]);
    const vec3<float> pos_world(p2.x, p2.y, 0.0f);
    const float angle = polygon_angle[item];
    const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0, 0, 1), angle);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.direction));
    vec3<float> ray_org_local = rotate(conj(q_world), vec3<float>(ray.origin) - pos_world);

    // find the point where the ray intersects the plane of the polygon
    const vec3<float> n = vec3<float>(0, 0, 1);
    const vec3<float> p = vec3<float>(0, 0, 0);

    float d = -dot(n, p);
    float denom = dot(n, ray_dir_local);
    float t_hit = -(d + dot(n, ray_org_local)) / denom;

    // if the ray is parallel to the plane, there is no intersection
    if (fabs(denom) < 1e-5)
    {
        return;
    }

    // see if the intersection point is inside the polygon
    vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;
    vec2<float> r_hit_2d(r_hit.x, r_hit.y);
    float d_edge, min_d;

    bool inside = is_inside(d_edge, r_hit_2d);

    // spheropolygon (equivalent to sharp polygon when rounding radius is 0
    // make distance signed (negative is inside)
    if (inside)
    {
        d_edge = -d_edge;
    }

    // exit if outside
    if (d_edge > polygon_rounding_radius)
    {
        return;
    }
    min_d = polygon_rounding_radius - d_edge;

    // if we get here, we hit the inside of the polygon
    if (rtPotentialIntersection(t_hit))
    {
        // make polygons double sided
        vec3<float> n_flip;
        if (dot(n, ray_dir_local) < 0.0f)
        {
            n_flip = n;
        }
        else
        {
            n_flip = -n;
        }

        shading_normal = rotate(q_world, n_flip);
        shading_color = RGB<float>(polygon_color[item]);
        shading_distance = min_d;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int item, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;

    vec2<float> p2 = vec2<float>(polygon_position[item]);
    vec3<float> p(p2.x, p2.y, 0.0f);

    aabb->m_min.x = p.x - polygon_radius;
    aabb->m_min.y = p.y - polygon_radius;
    aabb->m_min.z = p.z - 1e-5;

    aabb->m_max.x = p.x + polygon_radius;
    aabb->m_max.y = p.y + polygon_radius;
    aabb->m_max.z = p.z + 1e-5;
}
