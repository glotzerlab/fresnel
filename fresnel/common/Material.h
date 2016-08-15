// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "ColorMath.h"

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
    an output luminance based on input vectors.

    TODO: Document BRDF functions implemented here and what the parameters mean.
    TODO: Document linear vs SRGB color spaces.
*/
struct Material
    {
    float solid = 0;            //!< Set to 1 to pass through solid color
    colorRGB<float> color;      //!< Color of the material

    colorRGB<float> luminance()
        {
        if (solid > 0.5)
            return color;
        else
            return colorRGB<float>(0,0,0);
        }
    };

#undef DEVICE

#endif
