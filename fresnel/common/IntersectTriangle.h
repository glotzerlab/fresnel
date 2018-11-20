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

//! Ray triangle intersection test
/*!
\param u [out] Barycentric coordinate of intersection point
\param v [out] Barycentric coordinate of intersection point
\param w [out] Barycentric coordinate of intersection point
\param t [out] Intersection t value along ray
\param d_edge [out] Distance from shape edge in the view plane
\param p Start point of segment
\param q End point of segment
\param a Triangle vertex 0
\param b Triangle vertex 1
\param c Triangle vertex 2

From Real-time Collision Detection (Christer Ericson)
Given ray pq and triangle abc, returns whether segment intersects
triangle and if so, also returns the barycentric coordinates (u,v,w)
of the intersection point
Note: the triangle is assumed to be oriented counter-clockwise when viewed from the direction of p
*/
DEVICE inline bool intersect_ray_triangle(float &u, float &v, float &w, float &t, float &d_edge,
                                          vec3<float>& n, const vec3<float>& p, const vec3<float>& q,
                                          const vec3<float>& a, const vec3<float>& b, const vec3<float>& c)
    {
    vec3<float> ab = b - a;
    vec3<float> ac = c - a;
    vec3<float> qp = p - q;

    // Compute triangle normal. Can be precalculated or cached if
    // intersecting multiple segments against the same triangle
    n = cross(ab, ac);

    // Compute denominator d. If d <= 0, segment is parallel to or points
    // away from triangle, so exit early
    float d = dot(qp, n);
    if (d <= float(0.0)) return false;

    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    vec3<float> ap = p - a;
    t = dot(ap, n);
    if (t < float(0.0)) return false;
//    if (t > d) return false; // For segment; exclude this code line for a ray test

    // Compute barycentric coordinate components and test if within bounds
    vec3<float> e = cross(qp, ap);
    v = dot(ac, e);
    if (v < float(0.0) || v > d) return false;
    w = -dot(ab, e);
    if (w < float(0.0) || v + w > d) return false;

    // Segment/ray intersects triangle. Perform delayed division and
    // compute the last barycentric coordinate component
    float ood = float(1.0) / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = float(1.0) - v - w;

    // determine distance from the hit point to the nearest edge
    vec3<float> edge;
    vec3<float> pt;
    if (u < v)
        {
        if (u < w)
            {
            edge = c - b;
            pt = b;
            }
        else
            {
            edge =  b - a;
            pt = a;
            }
        }
    else
        {
        if (v < w)
            {
            edge = a - c;
            pt = c;
            }
        else
            {
            edge = b - a;
            pt = a;
            }
        }

    // find the distance from the hit point to the line
    vec3<float> r_hit = p + t * (q-p);
    vec3<float> vec = cross(edge, r_hit - pt);
    float dsq = dot(vec, vec) / dot(edge,edge);
    d_edge = sqrt(dsq);

    return true;
    }

}

#undef DEVICE

#endif
