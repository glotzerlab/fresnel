// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __INTERSECT_ELLIPSOID_H__
#define __INTERSECT_ELLIPSOID_H__

#include "common/VectorMath.h"
#include <cmath>

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel
    {
const float ellipsoid_epsilon = 1e-4f;

//! Ray-sphere intersection test
/*! \param t [out] Intersection t value along ray
    \param d_edge [out] Distance from shape edge in the view plane
    \param N [out] Normal vector
    \param o Ray origin
    \param d Ray direction (normalized)
    \param p Ellipsoid position
    \param r Ellipsoid radius

    \returns True if the ray intersects the sphere, False if it does not.

    Output arguments \a d and \a N are set when the intersection routine returns true.
    \a t may be set even if there is no intersection.
*/
DEVICE inline bool intersect_ray_ellipsoid(float& t,
                                        float& d_edge,
                                        vec3<float>& N,
                                        const vec3<float>& o,
                                        const vec3<float>& d,
                                        const vec3<float>& p,
										const vec3<float>& xyz,
                                        const float r)
    {
    // vector from ellipsoid to ray origin
    vec3<float> v = p - o;

	// replace v with ray_dir_local
	// replace o with ray_org_local
	
	  // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), dir); //rotates ray direction by conjugate of shape's quaternion
    vec3<float> ray_org_local
        = rotate(conj(q_world), vec3<float>(ray.org_x, ray.org_y, ray.org_z) - pos_world);
	
	// matrix to scale an ellipsoid with half lengths a,b,c into a unit sphere
	//M = {1/a, 0, 0}
	//    {0, 1/b, 0}
	//    {0, 0, 1/c}
	
	// scaled vector from ellipsoid to ray origin v'
	vec3<float> vp = ;// v*M
	// how to do matrix multiplication fast?

	// scaled ray direction d'
	vec3<float> dp ;
	 
    // solve intersection via quadratic formula
	float b = dot(vp, dp); // b = b' = v' dot d'
    float det = b * b - dot(vp, vp) + 1;

    // no solution when determinant is negative
    if (det < 0)
        return false;

    // the ray intersects the scaled ellipsoid, compute the distance in the viewing plane

	// not sure how to change this part for ellipsoid. Probably have to transform back?
    const vec3<float> w = cross(v, d);
    const float Dsq = dot(w, w); // assumes ray direction is normalized
    // The distance of the hit position from the edge of the sphere,
    // projected into the plane which has the ray as its normal
    d_edge = r - fast::sqrt(Dsq);

    // solve the quadratic equation
    det = fast::sqrt(det);

    // first case
    t = b - det;
    if (t > ellipsoid_epsilon)
        {
			N = o + t * d - p; //do we use N?
        return true;
        }

    // second case (origin is inside the sphere)
    t = b + det;
    if (t > ellipsoid_epsilon)
        {
        N = -(o + t * d - p);
        return true;
        }

    // both cases intersect the sphere behind the origin
    return false;
    }

    } // namespace fresnel

#undef DEVICE

#endif
