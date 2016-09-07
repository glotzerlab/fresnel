// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "VectorMath.h"

#ifndef __CAMERA_H__
#define __CAMERA_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

//! Camera properties
/*! Camera is a plain old data struct that holds camera properties, and a few methods for computing
    vectors in the image plane given normal screen coords.

    Some quantities in this struct are precomputed, normalized versions of user provided quantities.
*/
struct Camera
    {
    DEVICE Camera() {}
    Camera(const vec3<float>& _p,
                  const vec3<float>& _d,
                  const vec3<float>& _u,
                  float h)
        : p(_p), d(_d), u(_u), h(h)
        {
        // TODO: import fast:: math library from hoomd and use here

        // normalize inputs
        d *= 1.0f / sqrtf(dot(d, d));
        u *= 1.0f / sqrtf(dot(u, u));

        // form r
        r = cross(d,u);
        r *= 1.0f / sqrtf(dot(r, r));

        // make set orthonormal
        u = cross(r,d);
        u *= 1.0f / sqrtf(dot(u, u));
        }

    vec3<float> p;  //!< Center of the image plane
    vec3<float> d;  //!< Direction the camera faces (normalized)
    vec3<float> u;  //!< Up vector (normalized)
    vec3<float> r;  //!< Right vector (normalized)
    float h;        //!< Height of the camera image plane

    //! Get a ray start position given screen relative coordinates
    DEVICE vec3<float> origin(float xs, float ys) const
        {
        return p + (ys * u + xs * r) * h;
        }

    //! Get a ray direction given screen relative coordinates
    DEVICE vec3<float> direction(float xs, float ys) const
        {
        return d;
        }
    };

#undef DEVICE

#endif
