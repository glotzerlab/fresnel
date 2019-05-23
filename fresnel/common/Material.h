// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "ColorMath.h"
#include "VectorMath.h"

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef __CUDACC__
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

    Material implements Disney's principled BRDF. The evaluation is split into diffuse and specular terms to support
    representative point lights described in "Real Shading in Unreal Engine 4", SIGGRAPH course notes 2013. This method
    uses a different light direction for specular and diffuse terms and includes a normalization factor based on the
    size of the light source. It also implements a combined method for efficiency in the path tracer.

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
    float spec_trans;                    //!< Set to 0 for solid materials, 1 for fully transmissive

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
        float denom_rt = (1.0f + (alpha_sq - 1.0f)*ndoth*ndoth);
        float D = alpha_sq / (float(M_PI) * denom_rt*denom_rt);

        // F(theta_d)
        RGB<float> F0_dielectric(0.08f*specular, 0.08f*specular, 0.08f*specular);
        RGB<float> F0 = lerp(metal, F0_dielectric, base_color);
        RGB<float> F = F0 + (RGB<float>(1.0f, 1.0f, 1.0f) - F0)*schlick(ldoth);

        // G(theta_l, theta_v)
        // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
        // using disney's alpha_g remapping
        //float alpha_g = 0.5f + alpha / 2.0f;
        //float alpha_g_sq = alpha_g*alpha_g;
        float ndotv_sq = ndotv*ndotv;
        float ndotl_sq = ndotl*ndotl;
        float V1 = 1.0f / (ndotv + sqrtf(alpha_sq + ndotv_sq - alpha_sq * ndotv_sq));
        float V2 = 1.0f / (ndotl + sqrtf(alpha_sq + ndotl_sq - alpha_sq * ndotl_sq));
        float V = V1*V2;

        // the 4 cos(theta_l) cos(theta_v) factor is built into V
        RGB<float> f_s = D * F * V;

        return f_d * (1.0f - metal) + f_s;
        }


    DEVICE RGB<float> brdf_diffuse(vec3<float> l, vec3<float> v, vec3<float> n, const RGB<float>& shading_color) const
        {
        // diffuse BRDF is 0 when behind the surface or metal is 1.0f
        float ndotv = dot(n,v);     // cos(theta_v)
        if (ndotv <= 0 || metal == 1.0f)
            return RGB<float>(0,0,0);

        // compute h vector and cosines of relevant angles
        vec3<float> h = l+v;
        h /= sqrtf(dot(h,h));

        float ndotl = dot(n,l);     // cos(theta_l)
        float ldoth = dot(l,h);     // cos(theta_d)
        // float ndoth = dot(n,h);     // cos(theta_h)

        // precomputed parameters
        RGB<float> base_color = getColor(shading_color);

        // diffuse term (section 2.3 from Extending  the  Disney  BRDF  to  a  BSDF  with Integrated Subsurface Scattering)
        float FL = schlick(ndotl);
        float FV = schlick(ndotv);
        float RR = 2*roughness*ldoth*ldoth;
        RGB<float> f_d = base_color / float(M_PI) * ((1.0f - 0.5f * FL) * (1.0f - 0.5f * FV) +
                                                     RR * (FL + FV + FL*FV*(RR-1.0f)));

        return f_d * (1.0f - metal);
        }

    DEVICE RGB<float> brdf_specular(vec3<float> l, vec3<float> v, vec3<float> n, const RGB<float>& shading_color, float light_half_angle = 0.0f) const
        {
        // specular BRDF is 0 when behind the surface
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

        // specular term
        // D(theta_h) - using D_GTR_2 (eq 8 from Physically based Shading at Disney)
        float alpha = roughness*roughness;
        float alpha_sq = alpha*alpha;
        float denom_rt = (1.0f + (alpha_sq - 1.0f)*ndoth*ndoth);
        float D = alpha_sq / (float(M_PI) * denom_rt*denom_rt);
        float alpha_prime;

        // normalization correction per UE4 paper

        if (light_half_angle > 0.99f*float(M_PI)/2.0f)
            light_half_angle = 0.99f*float(M_PI)/2.0f;

        if (light_half_angle > 0.0f)
            {
            alpha_prime = alpha + 0.5f * tanf(light_half_angle);
            if (alpha_prime > 1.0f)
                alpha_prime = 1.0f;
            D = D*alpha_sq / (alpha_prime*alpha_prime);
            }

        // F(theta_d)
        RGB<float> F0_dielectric(0.08f*specular, 0.08f*specular, 0.08f*specular);
        RGB<float> F0 = lerp(metal, F0_dielectric, base_color);
        RGB<float> F = F0 + (RGB<float>(1.0f, 1.0f, 1.0f) - F0)*schlick(ldoth);

        // G(theta_l, theta_v)
        // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
        // using disney's alpha_g remapping
        //float alpha_g = 0.5f + alpha / 2.0f;
        //float alpha_g_sq = alpha_g*alpha_g;
        float ndotv_sq = ndotv*ndotv;
        float ndotl_sq = ndotl*ndotl;
        float V1 = 1.0f / (ndotv + sqrtf(alpha_sq + ndotv_sq - alpha_sq * ndotv_sq));
        float V2 = 1.0f / (ndotl + sqrtf(alpha_sq + ndotl_sq - alpha_sq * ndotl_sq));
        float V = V1*V2;

        // the 4 cos(theta_l) cos(theta_v) factor is built into V
        RGB<float> f_s = D * F * V;

        return f_s;
        }

    DEVICE bool isSolid() const
        {
        return (solid > 0.5f);
        }

    DEVICE RGB<float> getColor(const RGB<float>& shading_color) const
        {
        return lerp(primitive_color_mix, color, shading_color);
        }

    DEVICE vec3<float> importanceSampleGGX(vec2<float> xi, vec3<float> v, vec3<float> n) const
        {
        // use eq 9 from "Physically based shading at Disney" to compute the random direction to sample
        float alpha = roughness*roughness;
        float phi = 2.0f * float(M_PI) * xi.x;
        float cos_theta = sqrtf((1.0f - xi.y) / (1.0f + (alpha*alpha - 1.0f) * xi.y));
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        // put into vector form in the tangent space
        vec3<float> h_t(sin_theta * cosf(phi),
                        sin_theta * sinf(phi),
                        cos_theta);

        // convert tangent space to world space
        vec3<float> up(0,0,1.0f);
        if (fabs(n.z) > 0.999)
            up = vec3<float>(1.0f,0,0);
        vec3<float> t_x = cross(up, n);
        // TODO: normalize method in vectormath
        t_x = t_x / sqrtf(dot(t_x, t_x));
        vec3<float> t_y = cross(n, t_x);

        vec3<float> h = t_x * h_t.x + t_y * h_t.y + n * h_t.z;

        // convert from half vector to l vector
        vec3<float> l = 2.0f * dot(v, h) * h - v;
        return l;
        }

    DEVICE float pdfGGX(vec3<float> l, vec3<float> v, vec3<float> n) const
        {
        // compute h vector and cosines of relevant angles
        vec3<float> h = l+v;
        h /= sqrtf(dot(h,h));

        float ldoth = dot(l,h);     // cos(theta_d)
        float ndoth = dot(n,h);     // cos(theta_h)

        // D(theta_h) - using D_GTR_2 (eq 8 from Physically based Shading at Disney)
        float alpha = roughness*roughness;
        float alpha_sq = alpha*alpha;
        float denom_rt = (1.0f + (alpha_sq - 1.0f)*ndoth*ndoth);
        float D = alpha_sq / (float(M_PI) * denom_rt*denom_rt);

        // convert to pdf_l per equation in B.1 from "Physically based Shading at Disney"
        float pdf_l = D*ndoth / (4.0f * ldoth);

        if (pdf_l > 0.0f)
            return pdf_l;
        else
            return 0.0f;
        }

    DEVICE vec3<float> importanceSampleDiffuse(vec2<float> xi, vec3<float> v, vec3<float> n) const
        {
        const float r = sqrtf(xi.x);
        const float theta = 2 * float(M_PI) * xi.y;

        const float x = r * cosf(theta);
        const float y = r * sinf(theta);

        vec3<float> v_t(x, y, sqrt(1.0f - xi.x));

        // convert tangent space to world space
        vec3<float> up(0,0,1.0f);
        if (fabs(n.z) > 0.999)
            up = vec3<float>(1.0f,0,0);
        vec3<float> t_x = cross(up, n);
        // TODO: normalize method in vectormath
        t_x = t_x / sqrtf(dot(t_x, t_x));
        vec3<float> t_y = cross(n, t_x);

        vec3<float> l = t_x * v_t.x + t_y * v_t.y + n * v_t.z;

        return l;
        }

    DEVICE float pdfDiffuse(vec3<float> l, vec3<float> v, vec3<float> n) const
        {
        float ndotl = dot(n, l);
        if (ndotl > 0.0f)
            return ndotl / float(M_PI);
        else
            return 0.0f;
        }

    };

}

#undef DEVICE

#endif
