// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Random123/philox.h"
#include "uniform.hpp"
#include "boxmuller.hpp"

#include "VectorMath.h"
#include "Material.h"

#ifndef __RAYGEN_H__
#define __RAYGEN_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace fresnel {

// The following constants go in the 4th counter input to philox
// Each unique use of RNGs must use a different values to get different numbers.

//! Counter for anti-aliasing samples
const unsigned int rng_val_aa = 0x22ab5871;

//! Counter for ray path tracing samples (uniform sampling)
const unsigned int rng_val_uniform = 0x11ffabcd;

//! Counter for ray path tracing samples (multiple importance sampling)
const unsigned int rng_val_mis = 0x8754abcd;


//! Ray generation methods
/*! Common code to generate rays on the host and device.
*/
class RayGen
    {
    public:

        //! Default constructor gives uninitialized generator
        DEVICE RayGen() {}

        //! Set ray gen parameters
        DEVICE explicit RayGen(unsigned int i,
                               unsigned int j,
                               unsigned int width,
                               unsigned int height,
                               unsigned int seed) :
            m_width(width), m_height(height), m_i(i), m_j(j)
            {
            unsigned int pixel = j*width + i;

            // create the philox unique key for this RNG which includes the pixel ID and the random seed
            r123::Philox4x32::ukey_type rng_uk = {{pixel, seed}};
            m_rng_key = rng_uk;
            }

    //! Importance sample pixel locations for anti-aliasing
    /*! \param sample Index of the current sample

        Given the sample index, importance sample the tent filter to produce anti-aliased output.
    */
    DEVICE vec2<float> importanceSampleAA(unsigned int sample) const
        {
        // generate 2 random numbers from 0 to 2
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, 0, sample, rng_val_aa}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);
        float r1 = r123::u01<float>(rng_u[0]) * 2.0f;
        float r2 = r123::u01<float>(rng_u[1]) * 2.0f;

        // use important sampling to sample the tent filter
        float dx, dy;
        if (r1 < 1.0f)
            dx = sqrtf(r1) - 1.0f;
        else
            dx = 1.0f - sqrtf(2.0f - r1);

        if (r2 < 1.0f)
            dy = sqrtf(r2) - 1.0f;
        else
            dy = 1.0f - sqrtf(2.0f - r2);

        float i_f = float(m_i) + 0.5f + dx * m_aa_w;
        float j_f = float(m_j) + 0.5f + dy * m_aa_w;

        // determine the viewing plane relative coordinates
        float ys = -1.0f * (j_f / float(m_height) -0.5f);
        float xs = i_f / float(m_height) - 0.5f * float(m_width) / float(m_height);
        return vec2<float>(xs, ys);
        }

    //! Uniform sampling of reflected rays
    /*! \returns The direction to sample next
        \param factor [output] Weighting factor for the sample
        \param v Vector pointing back toward the viewing direction
        \param n Normal vector
        \param depth Depth of the ray in the trace
        \param sample Sample index
    */
    DEVICE vec3<float> uniformSampleReflection(float& factor,
                                               const vec3<float>& v,
                                               const vec3<float>& n,
                                               unsigned int depth,
                                               unsigned int sample
                                               ) const
        {
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, depth, sample, rng_val_uniform}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);

        // randomly pick a point on the sphere
        r123::float2 rng_gauss1 = r123::boxmuller(rng_u[0], rng_u[1]);
        r123::float2 rng_gauss2 = r123::boxmuller(rng_u[2], rng_u[3]);
        vec3<float> l(rng_gauss1.x, rng_gauss1.y, rng_gauss2.x);

        l = l / std::sqrt(dot(l, l));

        float ndotl = dot(n, l);
        // l is generated on the whole sphere, if it points down into the surface, make it point up
        if (ndotl < 0.0f)
            {
            l = -l;
            ndotl = -ndotl;
            }
        float pdf = 1.0f / (2.0f * float(M_PI));
        factor = 1.0f / pdf;
        return l;
        }


    //! Multiple importance sampling of reflected and transmitted rays
    /*! \returns The direction to sample next
        \param factor [output] Weighting factor for the sample
        \param transmit [output] True when transmission is selected, False for reflection
        \param v Vector pointing back toward the viewing direction
        \param n Normal vector
        \param depth Depth of the ray in the trace
        \param sample Sample index
        \param m Material
    */
    DEVICE vec3<float> MISReflectionTransmission(float& factor,
                                                 bool& transmit,
                                                 const vec3<float>& v,
                                                 const vec3<float>& n,
                                                 unsigned int depth,
                                                 unsigned int sample,
                                                 const Material& m
                                                 ) const
        {
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type rng_counter = {{0, depth, sample, rng_val_mis}};
        r123::Philox4x32::ctr_type rng_u = rng(rng_counter, m_rng_key);

        // multiple importance sampling
        vec2<float> xi(r123::u01<float>(rng_u[0]), r123::u01<float>(rng_u[1]));
        float choice_mis = r123::u01<float>(rng_u[2]);
        float choice_trans = r123::u01<float>(rng_u[3]);

        vec3<float> l;
        transmit = (choice_trans <= m.spec_trans);
        if (transmit)
            {
            // hard code perfect transmission
            l = -v;
            }
        else
            {
            // handle reflection with multiple importance sampling
            if (choice_mis <= 0.5f)
                {
                // diffuse sampling
                l = m.importanceSampleDiffuse(xi, v, n);
                float pdf_diffuse = m.pdfDiffuse(l, v, n);
                float pdf_ggx = m.pdfGGX(l, v, n);
                float w_diffuse = pdf_diffuse / (pdf_diffuse + pdf_ggx);
                factor = w_diffuse / (0.5f * pdf_diffuse);
                }
            else
                {
                // specular reflection
                l = m.importanceSampleGGX(xi, v, n);
                float pdf_diffuse = m.pdfDiffuse(l, v, n);
                float pdf_ggx = m.pdfGGX(l, v, n);
                float w_ggx = pdf_ggx / (pdf_diffuse + pdf_ggx);
                factor = w_ggx / (0.5f * pdf_ggx);
                }
            }
        return l;
        }

    protected:
        unsigned int m_width;                 //!< Width of the output image (in pixels)
        unsigned int m_height;                //!< Height of the output image (in pixels)
        const float m_aa_w = 0.707106781f;    //!< Width of the anti-aliasing filter (in pixels)
        r123::Philox4x32::key_type m_rng_key; //!< Key for the random number generator
        unsigned int m_i;                     //!< i coordinate of the pixel
        unsigned int m_j;                     //!< j coordinate of the pixel

    };

}
#undef DEVICE

#endif
