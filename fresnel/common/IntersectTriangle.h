// Copyright (c) 2016-2018 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __INTERSECT_Triangle_H__
#define __INTERSECT_Triangle_H__

#include "IntersectSphere.h"
#include <iostream>

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel {

const float mesh_epsilon = 1e-4f;

/*! From Real-time Collision Detection (Christer Ericson)
    Given ray do and triangle abc, returns whether segment intersects
    triangle and if so, also returns the barycentric coordinates (u,v,w)
    of the intersection point
    Note: the triangle is assumed to be oriented counter-clockwise when viewed from the direction of p
    \param t [out] Intersection t value along ray
    \param d_edge [out] Distance from shape edge in the view plane
    \param N [out] Normal vector
    \param q Ray origin
    \param p Ray direction (normalized)
    \param a mesh traingle vertex
    \param b mesh traingle vertex
    \param c mesh traingle vertex
    \param u [out] barycentric hit coordinate
    \param v [out] barycentric hit coordinate
    \param w [out] barycentric hit coordinate

    \returns True if the ray intersects the cylinder, False if it does not.

    Output arguments \a d_edge and \a N are set when the intersection routine returns true.
    \a t may be set even if there is no intersection.
*/

DEVICE inline bool intersect_ray_triangle(float& t,
                                          float& d_edge,
                                          vec3<float>& N,
                                          vec3<float>& color_index,
                                          const vec3<float>& q,
                                          const vec3<float>& p,
                                          const vec3<float>& a,
                                          const vec3<float>& b,
                                          const vec3<float>& c)
    {

    bool hit = false;
    vec3<float> ab = b - a;
    vec3<float> ac = c - a;
    vec3<float> pq = p - q;

    // Compute triangle normal. Can be precalculated or cached if
    // intersecting multiple segments against the same triangle
    vec3<float> n = cross(ab, ac);

    // Compute denominator d. If d <= 0, segment is parallel to or points
    // away from triangle, so exit early
    float d = dot(pq, n);
    if (d <= float(0.0)) hit=false;

    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    vec3<float> ap = p - a;
    t = dot(ap, n);
    if (t < float(0.0)) hit=false;
//    if (t > d) return false; // For segment; exclude this code line for a ray test

    // Compute barycentric coordinate components and test if within bounds
    vec3<float> e = cross(pq, ap);
    float v = dot(ac, e);
    if (v < float(0.0) || v > d) hit=false;
    float w = -dot(ab, e);
    if (w < float(0.0) || v + w > d) hit=false;

    // Segment/ray intersects triangle. Perform delayed division and
    // compute the last barycentric coordinate component
    float ood = float(1.0) / d;
    t *= ood;
    v *= ood;
    w *= ood;
    float u = float(1.0) - v - w;

    color_index.x=u;
    color_index.y=v;
    color_index.z=w;

    hit=true;

//        // determine distance from the hit point to the nearest edge
//        vec3<float> edge;
//        vec3<float> pt;
//        if (u < v)
//            {
//            if (u < w)
//                {
//                edge = v2 - v1;
//                pt = v1;
//                }
//            else
//                {
//                edge =  v1 - v0;
//                pt = v0;
//                }
//            }
//        else
//            {
//            if (v < w)
//                {
//                edge = v0 - v2;
//                pt = v2;
//                }
//            else
//                {
//                edge = v1 - v0;
//                pt = v0;
//                }
//            }
//
//        // find the distance from the hit point to the line
//        vec3<float> r_hit =  ray_org_local + t * ray_dir_local;
//        vec3<float> q = cross(edge, r_hit - pt);
//        float dsq = dot(q, q) / dot(edge,edge);
//        ray.d = sqrtf(dsq);


    return hit;
    }
}

#undef DEVICE

#endif
