// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef PLATFORM_H_
#define PLATFORM_H_

#include "common/ColorMath.h"
#include "common/VectorMath.h"
#include <limits>

#include <embree4/rtcore.h>

/*! Per the Embree documentation, this intersection context structure has the same data layout as
   the one in Embree's header, with extra custom bits at the end.

    The fresnel extensions are:

    - *d*: The distance to the nearest edge, provided by intersection routines
    - *shading_color*: The color of the primitive (or primitive subunit), provided by intersection
   routines
*/

struct FresnelRTCIntersectContext
    {
    FresnelRTCIntersectContext() : d(std::numeric_limits<float>::max()) { }

    RTCRayQueryContext context;

    // ray extensions go here
    float d; //!< Distance to the nearest edge
    fresnel::RGB<float> shading_color; //!< shading color determined by which primitive the ray hits
                                       //!< (or where on the primitive)
    };
#endif
