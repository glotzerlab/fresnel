// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __INTERSECT_SPHERE_H__
#define __INTERSECT_SPHERE_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

#include <cmath>

namespace fresnel {

const float sphere_epsilon = 1e-4f;

//! Ray-sphere intersection test
/*! \param t [out] Intersection t value along ray
    \param d_edge [out] Distance from shape edge in the view plane
    \param N [out] Normal vector
    \param o Ray origin
    \param d Ray direction (normalized)
    \param p Sphere position
    \param r Sphere radius

    \returns True if the ray intersects the sphere, False if it does not.

    Output arguments \a d and \a N are set when the intersection routine returns true.
    \a t may be set even if there is no intersection.
*/
DEVICE inline bool intersect_ray_sphere(float& t,
                                        float& d_edge,
                                        vec3<float>& N,
                                        const vec3<float>& o,
                                        const vec3<float>& d,
                                        const vec3<float>& p,
                                        const float r)
    {
    // vector from sphere to ray origin
    vec3<float> v = p-o;

    // solve intersection via quadratic formula
    float b = dot(v,d);
    float det = b*b - dot(v,v) + r*r;

    // no solution when determinant is negative
    if (det < 0)
        return false;

    // the ray intersects the sphere, compute the distance in the viewing plane
    const vec3<float> w = cross(v,d);
    const float Dsq = dot(w,w); // assumes ray direction is normalized
    // The distance of the hit position from the edge of the sphere,
    // projected into the plane which has the ray as it's normal
    d_edge = r - fast::sqrt(Dsq);

    // solve the quadratic equation
    det=fast::sqrt(det);

    // first case
    t = b - det;
    if (t > sphere_epsilon)
        {
        N = o+t*d-p;
        return true;
        }

    // second case (origin is inside the sphere)
    t = b + det;
    if (t > sphere_epsilon)
        {
        N = -(o+t*d-p);
        return true;
        }

    // both cases intersect the sphere behind the origin
    return false;
    }

}

#undef DEVICE

#endif
