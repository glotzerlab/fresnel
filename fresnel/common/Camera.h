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

    Some quantities in this struct are precomputed. The r vector must be the same length as u, and perpendicular
    to both u and d.
*/
struct Camera
    {
    Camera() {}
    Camera(vec3<float>& _p,
           vec3<float>& _d,
           vec3<float>& _u,
           vec3<float>& _r)
        : p(_p), d(_d), u(_u), r(_r)
        {
        }

    vec3<float> p;  //!< Center of the image plane
    vec3<float> d;  //!< Direction the camera faces
    vec3<float> u;  //!< Up vector
    vec3<float> r;  //!< Right vector

    //! Get a ray start position given screen relative coordinates
    vec3<float> origin(float xs, float ys) const
        {
        return p + ys*u + xs * r;
        }

    //! Get a ray direction given screen relative coordinates
    vec3<float> direction(float xs, float ys) const
        {
        return d;
        }
    };

#undef DEVICE

#endif
