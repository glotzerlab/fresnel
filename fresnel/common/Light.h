// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Camera.h"
#include "ColorMath.h"
#include "VectorMath.h"

#ifndef __LIGHT_H__
#define __LIGHT_H__

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
//! User Lights
/*! Store user provided light properties. These properties are convenient parameters for users to
   think about and set. The actual device camera properties are derived from these and stored in
   Lights.

    A single light is defined by a vector that points toward the light (in camera coordinates), a
   color, and the half angle of the area light.

    The Lights class stores up to 4 lights. They are stored in a fixed size plain old data struct
   for direct transfer to an OptiX variable.

    All of these parameters are directly modifiable as this is a plain old data structure.
*/
struct Lights
    {
    vec3<float> direction[4]; //!< Light directions
    RGB<float> color[4];      //!< Color of each light (linearized sRGB color space)
    float theta[4];           //!< Half angle of the area light
    unsigned int N;           //!< Number of lights

    //! Default constructor leaves memory uninitialized to support OptiX variables
    Lights() { }

    //! Put lights into scene coordinates given the camera
    explicit Lights(const Lights& lights, const Camera& cam)
        {
        N = lights.N;
        for (unsigned int i = 0; i < N; i++)
            {
            // copy over the color
            color[i] = lights.color[i];

            // normalize direction
            vec3<float> v = lights.direction[i];
            v *= 1.0f / sqrtf(dot(v, v));

            // put direction into scene coordinates
            const CameraBasis& basis = cam.getBasis();
            direction[i] = v.x * basis.u + v.y * basis.v + v.z * basis.w;

            // copy the angle
            theta[i] = lights.theta[i];
            }
        }

    vec3<float> getDirection(unsigned int i)
        {
        return direction[i];
        }

    void setDirection(unsigned int i, const vec3<float>& v)
        {
        direction[i] = v;
        }

    RGB<float> getColor(unsigned int i)
        {
        return color[i];
        }

    void setColor(unsigned int i, const RGB<float>& v)
        {
        color[i] = v;
        }

    float getTheta(unsigned int i)
        {
        return theta[i];
        }

    void setTheta(unsigned int i, float v)
        {
        theta[i] = v;
        }
    };

    } // namespace fresnel
#undef DEVICE

#endif
