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

namespace fresnel {

//! Compute schlick approximation to fresnel
/*! This function computes the (1-(cos(theta))**5 term only
    \param x cos(theta)
*/
DEVICE inline float schlick(float x)
    {
    float v = 1.0f - x;
    float v_sq = v*v;
    float v_fifth = v_sq * v_sq * v;
    return v_fifth;
    }

//! Material properties
/*! Material is a plain old data struct that holds material properties, and a few methods for computing
    an output brdf based on input vectors.

    Material currently implements a trivial diffuse BRDF.

    Material stores all colors in a linearized sRGB color space.
*/
struct Material
    {
    float solid;                         //!< Set to 1 to pass through solid color
    RGB<float> color;                    //!< Color of the material
    float primitive_color_mix;           //!< Set to 0 to force material color, 1 to use geometry color
    float roughness;                     //!< Set to 0 for a smooth material, non-zero for a rough material

    //! Default constructor gives uninitialized material
    DEVICE Material() {}

    //! Set material parameters
    DEVICE explicit Material(const RGB<float> _color, float _solid=0.0f) :
        solid(_solid), color(_color), primitive_color_mix(0.0f)
        {
        }

    DEVICE RGB<float> brdf(vec3<float> l, vec3<float> v, vec3<float> n, const RGB<float>& shading_color) const
        {
        // BRDF is 0 when behind the surface
        float ndotv = dot(n,v);
        if (ndotv <= 0)
            return RGB<float>(0,0,0);

        // compute h vector and cosines of relevant angles
        vec3<float> h = l+v;
        h /= sqrtf(dot(h,h));

        float ndotl = dot(n,l);
        float vdoth = dot(v,h);

        // precomputed parameters
        RGB<float> base_color = getColor(shading_color);
        float sigma = roughness*roughness;

        // diffuse term (section 5.3 from Physically based Shading at Disney)
        float FD90 = 0.5f + 2.0f * vdoth * vdoth * sigma;
        float f = (1.0f + (FD90 - 1.0f) * schlick(ndotl)) * (1.0f + (FD90 - 1.0f) * schlick(ndotv));
        if (f > 1.5)
            std::cout << f << std::endl;
        RGB<float> f_d = base_color / float(M_PI) * f;

        return f_d;
        }

    DEVICE bool isSolid() const
        {
        return (solid > 0.5f);
        }

    DEVICE RGB<float> getColor(const RGB<float>& shading_color) const
        {
        return lerp(primitive_color_mix, color, shading_color);
        }


    };

}

#undef DEVICE

#endif
