// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "VectorMath.h"
#include "Random123/philox.h"
#include "uniform.hpp"

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

//! Counter for ray path tracing samples
const unsigned int rng_val_ray = 0x11ffabcd;

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

    //! Important sample pixel locations for anti-aliasing
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
