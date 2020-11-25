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
    \param abc Ellipsoid radii a, b and c
	\param quat q_ellipsoid unit quaternion of the ellipsoid's orientation

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
										const vec3<float>& abc,
										const quat<float>& q_ellipsoid)
    {
    // vector from ellipsoid to ray origin
    vec3<float> v = p - o;

	// replace v with ray_dir_local
	// replace o with ray_org_local
	
	vec3<float> ray_dir_local = rotate(conj(q_ellipsoid), d);
	vec3<float> ray_origin_local = rotate(conj(q_ellipsoid), o);

	// scaled ray direction d'
	vec3<float> dp = ray_dir_local / abc;

	// normalize dp
	// TODO figure out how to optimize when to normalize and when not to
	float dp2 = dot(dp,dp);
	vec3<float> dpNorm = dp / fast::sqrt(dp2);

	// scaled ray origin o'
	vec3<float> op = ray_origin_local / abc;

	// scaled vector from ellipsoid to ray origin v' ("prime")
	//vec3<Real> vp = (o_c_local.x/a, o_c_local.y/b, o_c_local.z/c);
	vec3<float> vp = o_c_local / abc;
	
    // solve intersection via quadratic formula
	float b = dot(vp, dp); // b = b' = v' dot d'
    float det = b * b - dp2*dot(vp, vp) + dp2;

    // no solution when determinant is negative
    if (det < 0)
        return false;

    // the ray intersects the scaled ellipsoid, compute the distance in the viewing plane
    const vec3<float> wp = cross(vp, dpNorm);
    const float Dsq = dot(wp, wp); // assumes ray direction, d, is normalized
    // The distance of the hit position from the edge of the scaled ellipsoid,
    // projected into the plane which has the ray as its normal
    d_edge = 1 - fast::sqrt(Dsq);

	// TODO: transform back by multiplying by appropriate factors of a,b,c
	// d_edge = ;
	
    // solve the quadratic equation
    det = fast::sqrt(det);

    // first case
    t = (b - det) / dp2;
    if (t > ellipsoid_epsilon)
        {
			//N = o + t * d - p; // this just gives the radius of the sphere directed toward impact point
			const vec3<float> Np = op + t * dpNorm - p;
			// transform N back to world coordinates
			N = Np * abc;
			N = rotate(q_ellipsoid, Np);
        return true;
        }

    // second case (origin is inside the sphere)
    t = (b + det) / dp2;
    if (t > ellipsoid_epsilon)
        {
			// N = -(o + t * d - p);
			const vec3<float> Np = -(op + t * dpNorm - p);
			N = Np * abc;
			N = rotate(q_ellipsoid, Np);
        return true;
        }

    // both cases intersect the sphere behind the origin
    return false;
    }

    } // namespace fresnel

#undef DEVICE

#endif
