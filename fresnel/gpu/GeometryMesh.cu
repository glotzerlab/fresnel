// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include "common/ColorMath.h"
#include "common/IntersectTriangle.h"
#include "common/VectorMath.h"
#include <optix_world.h>

using namespace optix;
using namespace fresnel;

rtBuffer<float3> mesh_vertices;

rtBuffer<float3> mesh_position;
rtBuffer<float4> mesh_orientation;
rtBuffer<float3> mesh_color;

rtDeclareVariable(vec3<float>, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, shading_distance, attribute shading_distance, );
rtDeclareVariable(RGB<float>, shading_color, attribute shading_color, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
    {
    const unsigned int n_faces = mesh_vertices.size() / 3;
    const unsigned int i_poly = primIdx / n_faces;
    const unsigned int i_face = primIdx % n_faces;

    const vec3<float> p3(mesh_position[i_poly]);
    const quat<float> q_world(mesh_orientation[i_poly]);
    const vec3<float> ray_origin(ray.origin);
    const vec3<float> ray_direction(ray.direction);

    const vec3<float> v0(mesh_vertices[i_face * 3]);
    const vec3<float> v1(mesh_vertices[i_face * 3 + 1]);
    const vec3<float> v2(mesh_vertices[i_face * 3 + 2]);

    // transform the ray into the primitive coordinate system
    const vec3<float> ray_dir_local = rotate(conj(q_world), ray_direction);
    const vec3<float> ray_org_local = rotate(conj(q_world), ray_origin - p3);

    float t = 0, d = 0, u = 0, v = 0, w = 0;
    vec3<float> n;

    // double-sided triangle test
    if (!intersect_ray_triangle(u,
                                v,
                                w,
                                t,
                                d,
                                n,
                                ray_org_local,
                                ray_org_local + ray_dir_local,
                                v0,
                                v1,
                                v2)
        && !intersect_ray_triangle(v,
                                   u,
                                   w,
                                   t,
                                   d,
                                   n,
                                   ray_org_local,
                                   ray_org_local + ray_dir_local,
                                   v1,
                                   v0,
                                   v2))
        return;

    if (rtPotentialIntersection(t))
        {
        vec3<float> n_world = rotate(q_world, n);
        shading_normal = n_world;
        shading_distance = d;
        shading_color = RGB<float>(mesh_color[i_face * 3 + 0] * u + mesh_color[i_face * 3 + 1] * v
                                   + mesh_color[i_face * 3 + 2] * w);
        rtReportIntersection(0);
        }
    }

RT_PROGRAM void bounds(int primIdx, float result[6])
    {
    const unsigned int n_faces = mesh_vertices.size() / 3;
    const unsigned int i_poly = primIdx / n_faces;
    const unsigned int i_face = primIdx % n_faces;

    const vec3<float> p3(mesh_position[i_poly]);
    const quat<float> q_world(mesh_orientation[i_poly]);

    const vec3<float> v0(mesh_vertices[i_face * 3]);
    const vec3<float> v1(mesh_vertices[i_face * 3 + 1]);
    const vec3<float> v2(mesh_vertices[i_face * 3 + 2]);

    vec3<float> v0_world = rotate(q_world, v0) + p3;
    vec3<float> v1_world = rotate(q_world, v1) + p3;
    vec3<float> v2_world = rotate(q_world, v2) + p3;

    optix::Aabb* aabb = (optix::Aabb*)result;

    aabb->m_min.x = fminf(v0_world.x, fminf(v1_world.x, v2_world.x));
    aabb->m_min.y = fminf(v0_world.y, fminf(v1_world.y, v2_world.y));
    aabb->m_min.z = fminf(v0_world.z, fminf(v1_world.z, v2_world.z));
    aabb->m_max.x = fmaxf(v0_world.x, fmaxf(v1_world.x, v2_world.x));
    aabb->m_max.y = fmaxf(v0_world.y, fmaxf(v1_world.y, v2_world.y));
    aabb->m_max.z = fmaxf(v0_world.z, fmaxf(v1_world.z, v2_world.z));
    }
