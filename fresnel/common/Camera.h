// Copyright (c) 2016-2017 The Regents of the University of Michigan
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

namespace fresnel {

//! User Camera properties
/*! Store user provided camera properties. These properties are convenient parameters for users to think about
    and set. The actual device camera properties are derived from these and stored in Camera.

    The user provides the position of the center of projection, the point that camera looks at (which also defines
    the center of the focal plane), an up vector, and the height of the image plane.

    Later, perspective cameras will add a mode field, the focal length, and DOF cameras will add the aperture.

    All of these parameters are directly modifiable as this is a plain old data structure.
*/
struct UserCamera
    {
    vec3<float> position;
    vec3<float> look_at;
    vec3<float> up;
    float h;
    };


//! Store an orthonormal basis
/*! For a camera, an orthonormal basis is defined by a right (x), look (y) and up (z) directions.
*/
struct CameraBasis
    {
    CameraBasis() {}
    explicit CameraBasis(const UserCamera& user)
        : u(user.up)
        {
        d = user.look_at - user.position;

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

    vec3<float> d;  //!< Direction the camera faces (normalized)
    vec3<float> u;  //!< Up vector (orthonormal)
    vec3<float> r;  //!< Right vector (normalized)
    };

//! Device Camera properties
/*! Camera is a plain old data struct that holds camera properties, and a few methods for computing
    vectors in the image plane given normal screen coordinates. Normal screen coordinates range from
    -0.5 to 0.5 in the y direction and from -0.5*aspect to 0.5*aspect in the x direction, where aspect is the aspect
    ratio.

    A camera is defined by a position and coordinate system. The position  is the center of projection of
    the camera. The direction (normalized) is the direction that the camera points, right (normalized)
    points to the right and up (normalized) points up. The scalar *h* is the height of the image plane.
    The look_at position defines the point that the camera looks at and the center of the focal plane.
*/
struct Camera
    {
    Camera() {}
    explicit Camera(const UserCamera& user)
        : p(user.position), basis(user), h(user.h)
        {
        }

    vec3<float> p;     //!< Center of projection
    CameraBasis basis; //!< Camera coordinate basis
    float h;           //!< Height of the camera image plane

    //! Get a ray start position given screen normal coordinates
    DEVICE vec3<float> origin(const vec2<float>& s) const
        {
        return p + (s.y * basis.u + s.x * basis.r) * h;
        }

    //! Get a ray direction given screen relative coordinates
    DEVICE vec3<float> direction(const vec2<float>& s) const
        {
        return basis.d;
        }
    };

}

#undef DEVICE

#endif
