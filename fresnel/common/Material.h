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
    float specular;                      //!< Set to 0 for no specular highlights, 1 for strong highlights
    float metal;                         //!< Set to 0 for dielectric materials, set to 1 for metals

    //! Default constructor gives uninitialized material
    DEVICE Material() {}

    //! Set material parameters
    DEVICE explicit Material(const RGB<float> _color, float _solid=0.0f) :
        solid(_solid), color(_color), primitive_color_mix(0.0f), roughness(0.1f), specular(0.5f), metal(0.0f)
        {
        }

    DEVICE RGB<float> brdf(vec3<float> l, vec3<float> v, vec3<float> n, const RGB<float>& shading_color) const
        {
        // BRDF is 0 when behind the surface
        float ndotv = dot(n,v);     // cos(theta_v)
        if (ndotv <= 0)
            return RGB<float>(0,0,0);

        // compute h vector and cosines of relevant angles
        vec3<float> h = l+v;
        h /= sqrtf(dot(h,h));

        float ndotl = dot(n,l);     // cos(theta_l)
        float ldoth = dot(l,h);     // cos(theta_d)
        float ndoth = dot(n,h);     // cos(theta_h)

        // precomputed parameters
        RGB<float> base_color = getColor(shading_color);

        // diffuse term (section 2.3 from Extending  the  Disney  BRDF  to  a  BSDF  with Integrated Subsurface Scattering)
        float FL = schlick(ndotl);
        float FV = schlick(ndotv);
        float RR = 2*roughness*ldoth*ldoth;
        RGB<float> f_d = base_color / float(M_PI) * ((1.0f - 0.5f * FL) * (1.0f - 0.5f * FV) +
                                                     RR * (FL + FV + FL*FV*(RR-1.0f)));

        // specular term
        // D(theta_h) - using D_GTR_2 (eq 8 from Physically based Shading at Disney)
        float alpha = roughness*roughness;
        float alpha_sq = alpha*alpha;
        float denom_rt = (1 + (alpha_sq - 1)*ndoth*ndoth);
        float D = alpha_sq / (float(M_PI) * denom_rt*denom_rt);

        // F(theta_d)
        RGB<float> F0_dielectric(0.08f*specular, 0.08f*specular, 0.08f*specular);
        RGB<float> F0 = lerp(metal, F0_dielectric, color);
        RGB<float> F = F0 + (RGB<float>(1.0f, 1.0f, 1.0f) - F0)*schlick(ldoth);

        // G(theta_l, theta_v)
        // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

        // with roughness remapping by Disney
        // float roughness_remap = 0.5f+roughness/2.0f;
        // float alphag = roughness_remap*roughness_remap;
        // float alphag_sq = alphag*alphag;
        float ndotv_sq = ndotv*ndotv;
        float ndotl_sq = ndotl*ndotl;
        float V1 = 1.0f / (ndotv + sqrtf(alpha_sq + ndotv_sq - alpha_sq * ndotv_sq));
        float V2 = 1.0f / (ndotl + sqrtf(alpha_sq + ndotl_sq - alpha_sq * ndotl_sq));
        float V = V1*V2;

        // the 4 cos(theta_l) cos(theta_v) factor is built into V
        RGB<float> f_s = D * F * V;

        return f_d * (1.0f - metal) + f_s;
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
