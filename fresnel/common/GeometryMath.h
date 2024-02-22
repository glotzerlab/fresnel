// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOEMTRY_MATH_H__
#define GEOEMTRY_MATH_H__

#include "common/VectorMath.h"

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
//! Distance from point to line segment in 2d
/*! \param p point
    \param a Line end point
    \param b Line end point

    \returns The distance from the point to the line segment
*/
DEVICE inline float point_line_segment_distance(vec2<float> p, vec2<float> a, vec2<float> b)
    {
    // project p onto the line segment
    const vec2<float> ab = b - a;
    const float L = dot(ab, ab);
    const float t = fast::max(0.0f, fast::min(1.0f, dot(p - a, ab) / L));
    const vec2<float> projection = a + t * ab;

    // compute the distance between p and the projection
    const vec2<float> v = p - projection;
    return fast::sqrt(dot(v, v));
    }

    } // namespace fresnel

#undef DEVICE

#endif
