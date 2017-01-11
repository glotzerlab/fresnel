// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "ColorMath.h"
#include "VectorMath.h"

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

//! Material properties
/*! Material is a plain old data struct that holds material properties, and a few methods for computing
    an output brdf based on input vectors.

    TODO: Document BRDF functions implemented here and what the parameters mean.
    TODO: Document linear vs SRGB color spaces.
*/
struct Material
    {
    float solid = 0;                     //!< Set to 1 to pass through solid color
    RGB<float> color;                    //!< Color of the material
    float geometry_color_mix = 0.0f;     //!< Set to 0 to force material color, 1 to use geometry color

    DEVICE RGB<float> brdf(vec3<float> l, vec3<float> v, vec3<float> n, const RGB<float>& shading_color) const
        {
        // BRDF is 0 when behind the surface
        if (dot(n,v) <= 0)
            return RGB<float>(0,0,0);
        return getColor(shading_color) / float(M_PI);
        }

    DEVICE bool isSolid() const
        {
        return (solid > 0.5f);
        }

    DEVICE RGB<float> getColor(const RGB<float>& shading_color) const
        {
        return lerp(geometry_color_mix, color, shading_color);
        }


    };

#undef DEVICE

#endif
