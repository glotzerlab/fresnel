// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __INTERSECT_HEMISPHERE_H__
#define __INTERSECT_HEMISPHERE_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

#include <cmath>

namespace fresnel {

const float hemisphere_epsilon = 1e-4f;

DEVICE inline bool intersect_ray_plane(const vec3<float> &n, const vec3<float>& p0, const vec3<float> &l0, const vec3<float>& l, float &t)
    {
    // assuming vectors are all normalized
    float denom = dot(n, l);
    if (denom > hemisphere_epsilon)
        {
        vec3<float> p0l0 = p0 - l0;
        t = dot(p0l0, n) / denom;
        return (t >= 0);
        }

    return false;
    }

DEVICE inline bool intersect_ray_disk(const vec3<float> &n, const vec3<float> &p0, const float &radius, const vec3<float> &l0, const vec3<float> &l,
    float& t, float &d_edge)
    {
    if (intersect_ray_plane(n, p0, l0, l, t))
        {
        vec3<float> p = l0 + l * t;
        vec3<float> v = p - p0;
        float d2 = dot(v, v);
        d_edge = radius - fast::sqrt(d2);
        return d_edge > 0;
        // or you can use the following optimisation (and precompute radius^2)
        // return d2 <= radius2; // where radius2 = radius * radius
        }

     return false;
    }

//! Ray-sphere intersection test
/*! \param t [out] Intersection t value along ray
    \param d_edge [out] Distance from shape edge in the view plane
    \param N [out] Normal vector
    \param o Ray origin
    \param d Ray direction (normalized)
    \param p Hemisphere position
    \param r Hemisphere radius
    \param q Hemisphere director

    \returns True if the ray intersects the sphere, False if it does not.

    Output arguments \a d and \a N are set when the intersection routine returns true.
    \a t may be set even if there is no intersection.

    https://stackoverflow.com/questions/57134950/ray-tracing-a-hemisphere
*/
DEVICE inline bool intersect_ray_hemisphere(float& t,
                                        float& d_edge,
                                        vec3<float>& N,
                                        const vec3<float>& o,
                                        const vec3<float>& d,
                                        const vec3<float>& p,
                                        const float r,
                                        const vec3<float>& q)
    {
    // test intersection with hemisphere cap
    t = -1.0;

    float t_disk = -1.0;
    vec3<float> N_disk;
    float d0,t0;
    float d_edge_disk;
    if (intersect_ray_disk(q, p, r, o, d, t0, d0))
        {
        N_disk = q;
        t_disk = t0;
        d_edge_disk = d0;
        }

    if (intersect_ray_disk(-q, p, r, o, d, t0, d0))
        {
        N_disk = q;
        t_disk = t0;
        d_edge_disk = d0;
        }

    // vector from ray origin to sphere
    vec3<float> v = p-o;

    // solve intersection via quadratic formula
    float b = dot(v,d);
    float det = b*b - dot(v,v) + r*r;

    float t_sphere = -1.0;
    vec3<float> N_sphere;
    float d_edge_sphere = 0.0;

    // no solution on sphere when determinant is negative
    if (det >= 0)
        {
        // the ray intersects the sphere
        // solve the quadratic equation
        det=fast::sqrt(det);

        // first case
        float l0 = b - det;
        float l1 = b + det;

        // test both hits-p against normal q
        vec3<float> w = o+l0*d-p;
        if (dot(w,q) > hemisphere_epsilon)
            {
            // the ray hits the cut plane
            l0 = -1.0;
            }

        w = o+l1*d-p;
        if (dot(w,q) > hemisphere_epsilon)
            {
            l1 = -1.0;
            }

        if (l0<0.0)
            l0=l1;
        if (l1<0.0)
            l1=l0;

        t_sphere = l0 < l1 ? l0 : l1;
        if (t_sphere <= hemisphere_epsilon)
            t_sphere = l0 > l1 ? l0 : l1;

        if (t_sphere > hemisphere_epsilon)
            {
            N_sphere = o+t_sphere*d-p;
            d_edge_sphere = fast::sqrt(dot(N,N));
            }
        }

    t0 = t_sphere;
    float t1 = t_disk;

    vec3<float> N0 = N_sphere;
    vec3<float> N1 = N_disk;

    d0 = d_edge_sphere;
    float d1 = d_edge_disk;

    if (t0 < 0.0)
        {
        t = t1;
        N = N1;
        d_edge = d1;
        }

    if (t1 < 0.0)
        {
        t = t0;
        N = N0;
        d_edge = d0;
        }

    if (t0 >= 0.0 && t1 >= 0.0)
        {
        if (t0 < t1)
            {
            t = t0;
            N = N0;
            d_edge = d0;
            }
        else
            {
            t = t1;
            N = N1;
            d_edge = d1;
            }
        }

    return (t > 0);
    }

}

#undef DEVICE

#endif
