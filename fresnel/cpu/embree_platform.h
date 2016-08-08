#ifndef PLATFORM_H_
#define PLATFORM_H_

#include "common/VectorMath.h"
#include <limits>

#undef __noinline
#undef __forceinline
#define __noinline             __attribute__((noinline))
#define __forceinline          inline __attribute__((always_inline))
#define __RTCRay__

//! Custom ray structure.
/*! Per the Embree documentation, this ray structure has the same data layout as thye one in Embree's header,
    with extra custom bits at the end.
*/
struct alignas(16) RTCRay
    {
    //! Default constructor
    __forceinline RTCRay() {}

    //! Constructs a ray from origin, direction, and ray segment.
    __forceinline RTCRay(const vec3<float>& org,
                         const vec3<float>& dir,
                         float tnear = 0.0f,
                         float tfar = std::numeric_limits<float>::infinity(),
                         float time = 0.0f,
                         int mask = -1)
      : org(org), dir(dir), tnear(tnear), tfar(tfar), time(time), mask(mask), geomID(-1), primID(-1), instID(-1)
        {

        }

    //! Tests if we hit something.
    __forceinline bool hit() const
        {
        return geomID != -1;
        }

  public:
    vec3<float> org;     //!< Ray origin
    float _padding0;     //!< Padding to match embree RTCRay
    vec3<float> dir;     //!< Ray direction
    float _padding1;     //!< Padding to match embree RTCRay
    float tnear;         //!< Start of ray segment
    float tfar;          //!< End of ray segment
    float time;          //!< Time of this ray for motion blur.
    int mask;            //!< used to mask out objects during traversal

  public:
    vec3<float> Ng;      //!< Not normalized geometry normal
    float _padding2;     //!< Padding to match embree RTCRay
    float u;             //!< Barycentric u coordinate of hit
    float v;             //!< Barycentric v coordinate of hit
    int geomID;          //!< geometry ID
    int primID;          //!< primitive ID
    int instID;          //!< instance ID

  public:
    // ray extensions go here
    };


#endif
