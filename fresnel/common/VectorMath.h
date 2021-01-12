// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __VECTOR_MATH_H__
#define __VECTOR_MATH_H__

#include <cfloat>
#include <math.h>

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __CUDACC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

#ifndef __CUDA_ARCH__
#include <algorithm>
#endif

namespace fresnel
    {
/////////////////////////////// fast/slow math //////////////////////////////////

//! Fastmath routines
/*! Routines in the fast namespace map to fast math routines on the CPU and GPU. Where possible,
   these use the less accurate intrinsics on the GPU (i.e. __sinf). The routines are provide
   overloads for both single and double so that macro tricks aren't needed to handle single and
   double precision code.
*/
namespace fast
    {
//! Compute the reciprocal square root of x
inline DEVICE float rsqrt(float x)
    {
#ifdef __CUDA_ARCH__
    return ::rsqrtf(x);
#else
    return 1.0f / ::sqrtf(x);
#endif
    }

//! Compute the reciprocal square root of x
inline DEVICE double rsqrt(double x)
    {
#ifdef __CUDA_ARCH__
    return ::rsqrt(x);
#else
    return 1.0 / ::sqrt(x);
#endif
    }

//! Compute the sin of x
inline DEVICE float sin(float x)
    {
#ifdef __CUDA_ARCH__
    return __sinf(x);
#else
    return ::sinf(x);
#endif
    }

//! Compute the sin of x
inline DEVICE double sin(double x)
    {
    return ::sin(x);
    }

//! Compute the cos of x
inline DEVICE float cos(float x)
    {
#ifdef __CUDA_ARCH__
    return __cosf(x);
#else
    return ::cosf(x);
#endif
    }

//! Compute the cos of x
inline DEVICE double cos(double x)
    {
    return ::cos(x);
    }

//! Compute the pow of x,y
inline DEVICE float pow(float x, float y)
    {
#ifdef __CUDA_ARCH__
    return __powf(x, y);
#else
    return ::powf(x, y);
#endif
    }

//! Compute the sin of x
inline DEVICE double pow(double x, double y)
    {
    return ::pow(x, y);
    }

//! Compute the exp of x
inline DEVICE float exp(float x)
    {
#ifdef __CUDA_ARCH__
    return __expf(x);
#else
    return ::expf(x);
#endif
    }

//! Compute the exp of x
inline DEVICE double exp(double x)
    {
    return ::exp(x);
    }

//! Compute the sqrt of x
inline DEVICE float sqrt(float x)
    {
    return ::sqrtf(x);
    }

//! Compute the sqrt of x
inline DEVICE double sqrt(double x)
    {
    return ::sqrt(x);
    }

//! Compute the erfc of x
inline DEVICE float erfc(float x)
    {
    return ::erfcf(x);
    }

//! Compute the erfc of x
inline DEVICE double erfc(double x)
    {
    return ::erfc(x);
    }

//! Compute the acos of x
inline DEVICE float acos(float x)
    {
    return ::acosf(x);
    }

//! Compute the acos of x
inline DEVICE double acos(double x)
    {
    return ::acos(x);
    }

template<class Real> inline DEVICE Real min(Real a, Real b)
    {
#ifdef __CUDA_ARCH__
    return ::min(a, b);
#else
    return std::min(a, b);
#endif
    }

template<class Real> inline DEVICE Real max(Real a, Real b)
    {
#ifdef __CUDA_ARCH__
    return ::max(a, b);
#else
    return std::max(a, b);
#endif
    }
    } // namespace fast

//! Maximum accuracy math routines
/*! Routines in the slow namespace map to the most accurate version of the math routines on the CPU
   and GPU. The routines are provide overloads for both single and double so that macro tricks
   aren't needed to handle single and double precision code.

    These routines are intended to be used e.g. in integrators, where numerical stability is most
   important.
*/
namespace slow
    {
//! Compute the reciprocal square root of x
inline DEVICE float rsqrt(float x)
    {
#ifdef __CUDA_ARCH__
    return ::rsqrtf(x);
#else
    return 1.0f / ::sqrtf(x);
#endif
    }

//! Compute the reciprocal square root of x
inline DEVICE double rsqrt(double x)
    {
#ifdef __CUDA_ARCH__
    return ::rsqrt(x);
#else
    return 1.0 / ::sqrt(x);
#endif
    }

//! Compute the sin of x
inline DEVICE float sin(float x)
    {
#ifdef __CUDA_ARCH__
    return sinf(x);
#else
    return ::sinf(x);
#endif
    }

//! Compute the sin of x
inline DEVICE double sin(double x)
    {
    return ::sin(x);
    }

//! Compute the cos of x
inline DEVICE float cos(float x)
    {
#ifdef __CUDA_ARCH__
    return cosf(x);
#else
    return ::cosf(x);
#endif
    }

//! Compute the cos of x
inline DEVICE double cos(double x)
    {
    return ::cos(x);
    }

//! Compute the pow of x,y
inline DEVICE float pow(float x, float y)
    {
#ifdef __CUDA_ARCH__
    return powf(x, y);
#else
    return ::powf(x, y);
#endif
    }

//! Compute the sin of x
inline DEVICE double pow(double x, double y)
    {
    return ::pow(x, y);
    }

//! Compute the exp of x
inline DEVICE float exp(float x)
    {
#ifdef __CUDA_ARCH__
    return expf(x);
#else
    return ::expf(x);
#endif
    }

//! Compute the exp of x
inline DEVICE double exp(double x)
    {
    return ::exp(x);
    }

//! Compute the sqrt of x
inline DEVICE float sqrt(float x)
    {
    return ::sqrtf(x);
    }

//! Compute the sqrt of x
inline DEVICE double sqrt(double x)
    {
    return ::sqrt(x);
    }

//! Compute the erfc of x
inline DEVICE float erfc(float x)
    {
    return ::erfcf(x);
    }

//! Compute the erfc of x
inline DEVICE double erfc(double x)
    {
    return ::erfc(x);
    }

//! Compute the acos of x
inline DEVICE float acos(float x)
    {
    return ::acosf(x);
    }

//! Compute the acos of x
inline DEVICE double acos(double x)
    {
    return ::acos(x);
    }
    } // namespace slow

/////////////////////////////// vec3 ///////////////////////////////////

//! 3 element vector
/*! \tparam Real Data type of the components

    vec3 defines a simple 3 element vector. The components are available publicly as .x .y and .z.
   Along with vec3, a number of simple operations are defined to make writing vector math code
   easier. These include basic element-wise addition, subtraction, division, and multiplication (and
   += -= *= /=), and similarly division, and multiplication by scalars, and negation. The dot and
   cross product are also defined.

    \note Operations like normalize, length, etc... are purposefully **NOT** defined. These hide a
   sqrt. In high performance code, it is good for the programmer to have to explicitly call
   expensive operations.
*/
template<class Real> struct vec3
    {
    //! Construct a vec3
    /*! \param _x x-component
        \param _y y-component
        \param _z z-component
    */
    DEVICE vec3(const Real& _x, const Real& _y, const Real& _z) : x(_x), y(_y), z(_z) { }

    //! Construct an undefined vec3
    DEVICE vec3() { }

#ifdef __CUDACC__
    //! Convenience function to generate float3 in device code
    DEVICE operator float3()
        {
        return make_float3(x, y, z);
        }

    //! Convenience function to get a vec3 from a float3 in device code
    DEVICE explicit vec3(const float3& a) : x(a.x), y(a.y), z(a.z) { }
#endif

    Real x; //!< x-component of the vector
    Real y; //!< y-component of the vector
    Real z; //!< z-component of the vector
    };

//! Addition of two vec3s
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x+b.x, a.y+b.y, a.z+b.z).
*/
template<class Real> DEVICE inline vec3<Real> operator+(const vec3<Real>& a, const vec3<Real>& b)
    {
    return vec3<Real>(a.x + b.x, a.y + b.y, a.z + b.z);
    }

//! Subtraction of two vec3s
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.x-b.x, a.y-b.y, a.z-b.z).
*/
template<class Real> DEVICE inline vec3<Real> operator-(const vec3<Real>& a, const vec3<Real>& b)
    {
    return vec3<Real>(a.x - b.x, a.y - b.y, a.z - b.z);
    }

//! Multiplication of two vec3s
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.x*b.x, a.y*b.y, a.z*b.z).
*/
template<class Real> DEVICE inline vec3<Real> operator*(const vec3<Real>& a, const vec3<Real>& b)
    {
    return vec3<Real>(a.x * b.x, a.y * b.y, a.z * b.z);
    }

//! Division of two vec3s
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.x/b.x, a.y/b.y, a.z/b.z).
*/
template<class Real> DEVICE inline vec3<Real> operator/(const vec3<Real>& a, const vec3<Real>& b)
    {
    return vec3<Real>(a.x / b.x, a.y / b.y, a.z / b.z);
    }

//! Negation of a vec3
/*! \param a Vector

    Negation is component wise.
    \returns The vector (-a.x, -a.y, -a.z).
*/
template<class Real> DEVICE inline vec3<Real> operator-(const vec3<Real>& a)
    {
    return vec3<Real>(-a.x, -a.y, -a.z);
    }

//! Assignment-addition of two vec3s
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x += b.x, a.y += b.y, a.z += b.z).
*/
template<class Real> DEVICE inline vec3<Real>& operator+=(vec3<Real>& a, const vec3<Real>& b)
    {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
    }

//! Assignment-subtraction of two vec3s
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.x -= b.x, a.y -= b.y, a.z -= b.z).
*/
template<class Real> DEVICE inline vec3<Real>& operator-=(vec3<Real>& a, const vec3<Real>& b)
    {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
    }

//! Assignment-multiplication of two vec3s
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.x *= b.x, a.y *= b.y, a.z *= b.z).
*/
template<class Real> DEVICE inline vec3<Real>& operator*=(vec3<Real>& a, const vec3<Real>& b)
    {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
    }

//! Assignment-division of two vec3s
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.x /= b.x, a.y /= b.y, a.z /= b.z).
*/
template<class Real> DEVICE inline vec3<Real>& operator/=(vec3<Real>& a, const vec3<Real>& b)
    {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
    }

//! Multiplication of a vec3 by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x*b, a.y*b, a.z*b).
*/
template<class Real> DEVICE inline vec3<Real> operator*(const vec3<Real>& a, const Real& b)
    {
    return vec3<Real>(a.x * b, a.y * b, a.z * b);
    }

//! Multiplication of a vec3 by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x*b, a.y*b, a.z*b).
*/
template<class Real> DEVICE inline vec3<Real> operator*(const Real& b, const vec3<Real>& a)
    {
    return vec3<Real>(a.x * b, a.y * b, a.z * b);
    }

//! Division of a vec3 by a scalar
/*! \param a vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.x/b, a.y/b, a.z/b).
*/
template<class Real> DEVICE inline vec3<Real> operator/(const vec3<Real>& a, const Real& b)
    {
    Real q = Real(1.0) / b;
    return a * q;
    }

//! Assignment-multiplication of a vec3 by a scalar
/*! \param a First vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x *= b, a.y *= b, a.z *= b).
*/
template<class Real> DEVICE inline vec3<Real>& operator*=(vec3<Real>& a, const Real& b)
    {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
    }

//! Assignment-division of a vec3 by a scalar
/*! \param a First vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.x /= b, a.y /= b, a.z /= b).
*/
template<class Real> DEVICE inline vec3<Real>& operator/=(vec3<Real>& a, const Real& b)
    {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
    }

//! Equality test of two vec3s
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template<class Real> DEVICE inline bool operator==(const vec3<Real>& a, const vec3<Real>& b)
    {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
    }

//! Inequality test of two vec3s
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are not identically equal, and false if they are
*/
template<class Real> DEVICE inline bool operator!=(const vec3<Real>& a, const vec3<Real>& b)
    {
    return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
    }

//! dot product of two vec3s
/*! \param a First vector
    \param b Second vector

    \returns the dot product a.x*b.x + a.y*b.y + a.z*b.z.
*/
template<class Real> DEVICE inline Real dot(const vec3<Real>& a, const vec3<Real>& b)
    {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
    }

//! cross product of two vec3s
/*! \param a First vector
    \param b Second vector

    \returns the cross product (a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y *
   b.x).
*/
template<class Real> DEVICE inline vec3<Real> cross(const vec3<Real>& a, const vec3<Real>& b)
    {
    return vec3<Real>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

/////////////////////////////// vec2 ///////////////////////////////////

//! 2 element vector
/*! \tparam Real Data type of the components

    vec2 defines a simple 2 element vector. The components are available publicly as .x and .y.
   Along with vec2, a number of simple operations are defined to make writing vector math code
   easier. These include basic element-wise addition, subtraction, division, and multiplication (and
   += -= *= /=), and similarly division, and multiplication by scalars, and negation. The dot
   product is also defined.

    \note Operations like normalize, length, etc... are purposefully **NOT** defined. These hide a
   sqrt. In high performance code, it is good for the programmer to have to explicitly call
   expensive operations.
*/
template<class Real> struct vec2
    {
    //! Construct a vec2
    /*! \param _x x-component
        \param _y y-component
    */
    DEVICE vec2(const Real& _x, const Real& _y) : x(_x), y(_y) { }

    //! Default construct an undefined vector
    DEVICE vec2() { }

#ifdef __CUDACC__
    //! Convenience function to generate float2 in device code
    DEVICE operator float2()
        {
        return make_float2(x, y);
        }

    //! Convenience function to get a vec2 from a float2 in device code
    DEVICE explicit vec2(const float2& a) : x(a.x), y(a.y) { }
#endif

    //! Swap with another vector
    DEVICE void swap(vec2<Real>& v)
        {
        Real tx, ty;
        tx = v.x;
        ty = v.y;
        v.x = x;
        v.y = y;
        x = tx;
        y = ty;
        }

    Real x; //!< x-component of the vector
    Real y; //!< y-component of the vector
    };

//! Addition of two vec2s
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x+b.x, a.y+b.y).
*/
template<class Real> DEVICE inline vec2<Real> operator+(const vec2<Real>& a, const vec2<Real>& b)
    {
    return vec2<Real>(a.x + b.x, a.y + b.y);
    }

//! Subtraction of two vec2s
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.x-b.x, a.y-b.y).
*/
template<class Real> DEVICE inline vec2<Real> operator-(const vec2<Real>& a, const vec2<Real>& b)
    {
    return vec2<Real>(a.x - b.x, a.y - b.y);
    }

//! Multiplication of two vec2s
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.x*b.x, a.y*b.y).
*/
template<class Real> DEVICE inline vec2<Real> operator*(const vec2<Real>& a, const vec2<Real>& b)
    {
    return vec2<Real>(a.x * b.x, a.y * b.y);
    }

//! Division of two vec2s
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.x/b.x, a.y/b.y).
*/
template<class Real> DEVICE inline vec2<Real> operator/(const vec2<Real>& a, const vec2<Real>& b)
    {
    return vec2<Real>(a.x / b.x, a.y / b.y);
    }

//! Negation of a vec2
/*! \param a Vector

    Negation is component wise.
    \returns The vector (-a.x, -a.y).
*/
template<class Real> DEVICE inline vec2<Real> operator-(const vec2<Real>& a)
    {
    return vec2<Real>(-a.x, -a.y);
    }

//! Assignment-addition of two vec2s
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x += b.x, a.y += b.y).
*/
template<class Real> DEVICE inline vec2<Real>& operator+=(vec2<Real>& a, const vec2<Real>& b)
    {
    a.x += b.x;
    a.y += b.y;
    return a;
    }

//! Assignment-subtraction of two vec2s
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.x -= b.x, a.y -= b.y).
*/
template<class Real> DEVICE inline vec2<Real>& operator-=(vec2<Real>& a, const vec2<Real>& b)
    {
    a.x -= b.x;
    a.y -= b.y;
    return a;
    }

//! Assignment-multiplication of two vec2s
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.x *= b.x, a.y *= b.y).
*/
template<class Real> DEVICE inline vec2<Real>& operator*=(vec2<Real>& a, const vec2<Real>& b)
    {
    a.x *= b.x;
    a.y *= b.y;
    return a;
    }

//! Assignment-division of two vec2s
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.x /= b.x, a.y /= b.y).
*/
template<class Real> DEVICE inline vec2<Real>& operator/=(vec2<Real>& a, const vec2<Real>& b)
    {
    a.x /= b.x;
    a.y /= b.y;
    return a;
    }

//! Multiplication of a vec2 by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x*b, a.y*b).
*/
template<class Real> DEVICE inline vec2<Real> operator*(const vec2<Real>& a, const Real& b)
    {
    return vec2<Real>(a.x * b, a.y * b);
    }

//! Multiplication of a vec2 by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x*b, a.y*b).
*/
template<class Real> DEVICE inline vec2<Real> operator*(const Real& b, const vec2<Real>& a)
    {
    return vec2<Real>(a.x * b, a.y * b);
    }

//! Division of a vec2 by a scalar
/*! \param a vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.x/b, a.y/b, a.z/b).
*/
template<class Real> DEVICE inline vec2<Real> operator/(const vec2<Real>& a, const Real& b)
    {
    Real q = Real(1.0) / b;
    return a * q;
    }

//! Assignment-multiplication of a vec2 by a scalar
/*! \param a First vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x *= b, a.y *= b).
*/
template<class Real> DEVICE inline vec2<Real>& operator*=(vec2<Real>& a, const Real& b)
    {
    a.x *= b;
    a.y *= b;
    return a;
    }

//! Assignment-division of a vec2 by a scalar
/*! \param a First vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.x /= b, a.y /= b).
*/
template<class Real> DEVICE inline vec2<Real>& operator/=(vec2<Real>& a, const Real& b)
    {
    a.x /= b;
    a.y /= b;
    return a;
    }

//! Equality test of two vec2s
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template<class Real> DEVICE inline bool operator==(const vec2<Real>& a, const vec2<Real>& b)
    {
    return (a.x == b.x) && (a.y == b.y);
    }

//! Inequality test of two vec2s
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are not identically equal, false if they are
*/
template<class Real> DEVICE inline bool operator!=(const vec2<Real>& a, const vec2<Real>& b)
    {
    return (a.x != b.x) || (a.y != b.y);
    }

//! dot product of two vec2s
/*! \param a First vector
    \param b Second vector

    \returns the dot product a.x*b.x + a.y*b.y.
*/
template<class Real> DEVICE inline Real dot(const vec2<Real>& a, const vec2<Real>& b)
    {
    return (a.x * b.x + a.y * b.y);
    }

//! vec2 perpendicular operation
/*! \param a vector
    \returns a vector perpendicular to *a* (a.y, -a.x)
*/
template<class Real> DEVICE inline vec2<Real> perp(const vec2<Real>& a)
    {
    return vec2<Real>(-a.y, a.x);
    }

//! vec2 perpendicular dot product
/*! \param a first vector
    \param b second vector
    \returns the perpendicular dot product of a and b
*/
template<class Real> DEVICE inline Real perpdot(const vec2<Real>& a, const vec2<Real>& b)
    {
    return dot(perp(a), b);
    }

/////////////////////////////// quat ///////////////////////////////////

//! Quaternion
/*! \tparam Real Data type of the components

    quat defines a quaternion. The standard representation
   (https://en.wikipedia.org/wiki/Quaternion) is a1 + bi + cj + dk, or as a 4-vector (a, b, c, d). A
   more compact an expressive representation is to use a scalar and a 3-vector (s, v) where v =
   (b,c,d). This is the representation that quat uses.

    The following operators are defined for quaternions.
        - quat * scalar
        - scalar * quat
        - quat + quat
        - quat += quat
        - quat * quat
        - quat * vec3 (vec3 is promoted to quaternion (0,v))
        - vec3 * quat (vec3 is promoted to quaternion (0,v))
        - norm2(quat)
        - conj(quat)
        - rotate(quat, vec3)

    For more info on this representation and its relation to rotation, see:
    http://people.csail.mit.edu/bkph/articles/Quaternions.pdf


    \note normalize is purposefully **NOT** defined. It would hide hide a sqrt. In high
    performance code, it is good for the programmer to have to explicitly call expensive operations.
   In lieu of this, norm2 is provided, which computes the norm of a quaternion, squared.
*/
template<class Real> struct quat
    {
    //! Construct a quaternion
    /*! \param _s scalar component
        \param _v vector component
    */
    DEVICE quat(const Real& _s, const vec3<Real>& _v) : s(_s), v(_v) { }

    //! Default construct a unit quaternion
    DEVICE quat() : s(1), v(vec3<Real>(0, 0, 0)) { }

#ifdef __CUDACC__
    //! Convenience function to get a quat from a float4 in device code
    DEVICE explicit quat(const float4& a) : s(a.x), v(vec3<float>(a.y, a.z, a.w)) { }
#endif

    //! Construct a quat from an axis and an angle.
    /*! \param axis angle to represent
        \param theta angle to represent

        This is a convenience function for easy initialization of rotmat3s from an axis and an
       angle. The rotmat3 will initialize to the same rotation as the angle around the specified
       axis.
    */
    DEVICE static quat fromAxisAngle(const vec3<Real>& axis, const Real& theta)
        {
        quat<Real> q(fast::cos(theta / Real(2.0)), (Real)fast::sin(theta / Real(2.0)) * axis);
        return q;
        }

    Real s;       //!< scalar component
    vec3<Real> v; //!< vector component
    };

//! Multiplication of a quat by a scalar
/*! \param b scalar
    \param a quat

    Multiplication is component wise.
    \returns The quaternion (b*a.s, b*a.v).
*/
template<class Real> DEVICE inline quat<Real> operator*(const Real& b, const quat<Real>& a)
    {
    return quat<Real>(b * a.s, b * a.v);
    }

//! Multiplication of a quat by a scalar
/*! \param a quat
    \param b scalar

    Multiplication is component wise.
    \returns The quaternion (a.s*b, a.v*b).
*/
template<class Real> DEVICE inline quat<Real> operator*(const quat<Real>& a, const Real& b)
    {
    return quat<Real>(a.s * b, a.v * b);
    }

//! Addition of two quats
/*! \param a First quat
    \param b Second quat

    Addition is component wise.
    \returns The quaternion (a.s + b.s, a.v+b.v).
*/
template<class Real> DEVICE inline quat<Real> operator+(const quat<Real>& a, const quat<Real>& b)
    {
    return quat<Real>(a.s + b.s, a.v + b.v);
    }

//! Assignment-addition of two quats
/*! \param a First quat
    \param b Second quat

    Addition is component wise.
    \returns The quaternion (a.s += b.s, a.v += b.v).
*/
template<class Real> DEVICE inline quat<Real>& operator+=(quat<Real>& a, const quat<Real>& b)
    {
    a.s += b.s;
    a.v += b.v;
    return a;
    }

//! Subtraction of two quats
/*! \param a First quat
    \param b Second quat

    Subtraction is component wise.
    \returns The quaternion (a.s - b.s, a.v - b.v).
*/
template<class Real> DEVICE inline quat<Real> operator-(const quat<Real>& a, const quat<Real>& b)
    {
    return quat<Real>(a.s - b.s, a.v - b.v);
    }

//! Assignment-addition of two quats
/*! \param a First quat
    \param b Second quat

    Subtraction is component wise.
    \returns The quaternion (a.s -= b.s, a.v -= b.v).
*/
template<class Real> DEVICE inline quat<Real>& operator-=(quat<Real>& a, const quat<Real>& b)
    {
    a.s -= b.s;
    a.v -= b.v;
    return a;
    }

//! Multiplication of two quats
/*! \param a First quat
    \param b Second quat

    Multiplication is quaternion multiplication, defined as the cross product minus the dot product.
    (Ref. http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternions_briefly)
    When quaternions are being used for rotation, the composition of two rotation operations can be
    replaced by the quaternion product of the second rotation quaternion times the first.
    Note that quaternion multiplication is non-commutative.
    \returns The quaternion
    (a.s * b.s − dot(a.v, b.v), a.s*b.v + b.s * a.v + cross(a.v, b.v)).
*/
template<class Real> DEVICE inline quat<Real> operator*(const quat<Real>& a, const quat<Real>& b)
    {
    return quat<Real>(a.s * b.s - dot(a.v, b.v), a.s * b.v + b.s * a.v + cross(a.v, b.v));
    }

//! Multiplication of a vector by a quaternion
/*! \param a vector
    \param b quat

    Multiplication is quaternion multiplication. The vector is promoted to a quaternion (0,a)

    \returns The quaternion (a.s * b.s − dot(a.v, b.v), a.s*b.v + b.s * a.v + cross(a.v, b.v)).
*/
template<class Real> DEVICE inline quat<Real> operator*(const vec3<Real>& a, const quat<Real>& b)
    {
    return quat<Real>(0, a) * b;
    }

//! Multiplication of a quaternion by a vector
/*! \param a quat
    \param b vector

    Multiplication is quaternion multiplication. The vector is promoted to a quaternion (0,b)

    \returns The quaternion (a.s * b.s − dot(a.v, b.v), a.s*b.v + b.s * a.v + cross(a.v, b.v)).
*/
template<class Real> DEVICE inline quat<Real> operator*(const quat<Real>& a, const vec3<Real>& b)
    {
    return a * quat<Real>(0, b);
    }

//! norm squared of a quaternion
/*! \param a quat

    \returns the norm of the quaternion, squared. (a.s*a.s + dot(a.v,a.v))
*/
template<class Real> DEVICE inline Real norm2(const quat<Real>& a)
    {
    return (a.s * a.s + dot(a.v, a.v));
    }

//! conjugate of a quaternion
/*! \param a quat

    \returns the conjugate of the quaternion. (a.s, -a.v)
*/
template<class Real> DEVICE inline quat<Real> conj(const quat<Real>& a)
    {
    return quat<Real>(a.s, -a.v);
    }

//! rotate a vec3 by a quaternion
/*! \param a quat (should be a unit quaternion (Cos(theta/2), Sin(theta/2)*axis_unit_vector))
    \param b vector to rotate

    \returns the vector rotated by the quaternion, equivalent to the vector component of
   a*b*conj(a);
*/
template<class Real> DEVICE inline vec3<Real> rotate(const quat<Real>& a, const vec3<Real>& b)
    {
    // quat<Real> result = a*b*conj(a);
    // return result.v;

    // note: this version below probably results in fewer math operations. Need to double check that
    // it works when testing. I guesstimate only 20 clock ticks to rotate a vector with this code.
    // it comes from http://people.csail.mit.edu/bkph/articles/Quaternions.pdf
    return (a.s * a.s - dot(a.v, a.v)) * b + 2 * a.s * cross(a.v, b) + 2 * dot(a.v, b) * a.v;

    // from http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    // a suggested method for 15 mults and 15 adds, also in the above pdf:
    // return b + cross( (Real(2) * a.v), (cross(a.v,b) + (a.s * b)) );
    }

//! rotate a vec2 by a quaternion
/*! \param a quat (should be a unit quaternion (Cos(theta/2), Sin(theta/2)*axis_unit_vector))
    \param b vector to rotate

    \returns the vector rotated by the quaternion, equivalent to the vector component of
   a*b*conj(a);

    *b* is promoted to a 3d vector with z=0 for the rotation.
*/
template<class Real> DEVICE inline vec2<Real> rotate(const quat<Real>& a, const vec2<Real>& b)
    {
    vec3<Real> b3(b.x, b.y, Real(0.0));
    // b3 = (a*b3*conj(a)).v;

    // note: this version below probably results in fewer math operations. Need to double check that
    // it works when testing. I guesstimate only 20 clock ticks to rotate a vector with this code.
    // it comes from http://people.csail.mit.edu/bkph/articles/Quaternions.pdf
    b3 = (a.s * a.s - dot(a.v, a.v)) * b3 + 2 * a.s * cross(a.v, b3) + 2 * dot(a.v, b3) * a.v;

    // from http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    // a suggested method for 15 mults and 15 adds, also in the above pdf:
    // b3 = b3 + cross( (Real(2) * a.v), (cross(a.v,b3) + (a.s * b3)) );
    return vec2<Real>(b3.x, b3.y);
    }

//! dot product of two quats
/*! \param a First quat
    \param b Second quat

    \returns the dot product a.s*b.s+a.v.x*b.v.x + a.v.y*b.v.y + a.v.z*b.v.z.
*/
template<class Real> DEVICE inline Real dot(const quat<Real>& a, const quat<Real>& b)
    {
    return (a.s * b.s + dot(a.v, b.v));
    }

/////////////////////////////// rotmat2 ////////////////////////////////

//! 2x2 rotation matrix
/*! This is not a general 2x2 matrix class, but is specific to rotation matrices. It is designed for
   the use case where the same quaternion is repeatedly applied to many vectors in an inner loop.
   Using a rotation matrix decreases the number of FLOPs needed at the (potential) cost of increased
   register pressure. Both methods should be benchmarked to evaluate the actual performance
   trade-offs.

    The following operators are defined for rotmat2s.
        - rotmat2 * vec2
        - rotmat2(quat) - *constructs from a quaternion*
*/
template<class Real> struct rotmat2
    {
    //! Construct a quaternion
    /*! \param _row0 First row
        \param _row1 Second row
    */
    DEVICE rotmat2(const vec2<Real>& _row0, const vec2<Real>& _row1) : row0(_row0), row1(_row1) { }

    //! Construct a rotmat2 from a quat
    /*! \param q quaternion to represent

        This is a convenience function for easy initialization of rotmat2s from quats. The rotmat2
       will initialize to the same rotation as the quaternion.

        formula from http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    */
    DEVICE explicit rotmat2(const quat<Real>& q)
        {
        Real a = q.s, b = q.v.x, c = q.v.y, d = q.v.z;

        row0.x = a * a + b * b - c * c - d * d;
        row0.y = 2 * b * c - 2 * a * d;
        row1.x = 2 * b * c + 2 * a * d;
        row1.y = a * a - b * b + c * c - d * d;
        }

    //! Default construct an identity matrix
    DEVICE rotmat2() : row0(vec2<Real>(1, 0)), row1(vec2<Real>(0, 1)) { }

    //! Construct a rotmat2 from a float. formula from http://en.wikipedia.org/wiki/Rotation_matrix
    /*! \param theta angle to represent

        This is a convenience function for easy initialization of rotmat2s from angles. The rotmat2
       will initialize to the same rotation as the angle.
    */
    DEVICE static rotmat2 fromAngle(const Real& theta)
        {
        vec2<Real> row0;
        vec2<Real> row1;
        row0.x = fast::cos(theta);
        row0.y = -fast::sin(theta);
        row1.x = fast::sin(theta);
        row1.y = fast::cos(theta);
        return rotmat2<Real>(row0, row1);
        }

    vec2<Real> row0; //!< First row
    vec2<Real> row1; //!< Second row
    };

//! Matrix vector multiplication
/*! \param A matrix
    \param b vector
    \returns A*b

    Multiplication is matrix multiplication, where the vector is represented as a column vector.
*/
template<class Real> DEVICE inline vec2<Real> operator*(const rotmat2<Real>& A, const vec2<Real>& b)
    {
    return vec2<Real>(dot(A.row0, b), dot(A.row1, b));
    }

//! Transpose a rotmat2
/*! \param A matrix
    \returns the transpose of A

    A rotation matrix has an inverse equal to its transpose. There may be times where an algorithm
   needs to undo a rotation, so the transpose method is provided.
*/
template<class Real> DEVICE inline rotmat2<Real> transpose(const rotmat2<Real>& A)
    {
    return rotmat2<Real>(vec2<Real>(A.row0.x, A.row1.x), vec2<Real>(A.row0.y, A.row1.y));
    }

/////////////////////////////// rotmat3 ////////////////////////////////

//! 3x3 rotation matrix
/*! This is not a general 3x3 matrix class, but is specific to rotation matrices. It is designed for
   the use case where the same quaternion is repeatedly applied to many vectors in an inner loop.
   Using a rotation matrix decreases the number of FLOPs needed at the (potential) cost of increased
   register pressure. Both methods should be benchmarked to evaluate the actual performance
   trade-offs.

    The following operators are defined for rotmat3s.
        - rotmat3 * vec3
        - rotmat3(quat) - *constructs from a quaternion*

    \note Do not yet depend on the internal representation by rows. Future versions may store data
   in a different way.
*/
template<class Real> struct rotmat3
    {
    //! Construct a quaternion
    /*! \param _row0 First row
        \param _row1 Second row
        \param _row2 Third row
    */
    DEVICE rotmat3(const vec3<Real>& _row0, const vec3<Real>& _row1, const vec3<Real>& _row2)
        : row0(_row0), row1(_row1), row2(_row2)
        {
        }

    //! Construct a rotmat3 from a quat
    /*! \param q quaternion to represent

        This is a convenience function for easy initialization of rotmat3s from quats. The rotmat3
       will initialize to the same rotation as the quaternion.
    */
    DEVICE explicit rotmat3(const quat<Real>& q)
        {
        // formula from http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        Real a = q.s, b = q.v.x, c = q.v.y, d = q.v.z;

        row0.x = a * a + b * b - c * c - d * d;
        row0.y = 2 * b * c - 2 * a * d;
        row0.z = 2 * b * d + 2 * a * c;

        row1.x = 2 * b * c + 2 * a * d;
        row1.y = a * a - b * b + c * c - d * d;
        row1.z = 2 * c * d - 2 * a * b;

        row2.x = 2 * b * d - 2 * a * c;
        row2.y = 2 * c * d + 2 * a * b;
        row2.z = a * a - b * b - c * c + d * d;
        }

    //! Default construct an identity matrix
    DEVICE rotmat3()
        : row0(vec3<Real>(1, 0, 0)), row1(vec3<Real>(0, 1, 0)), row2(vec3<Real>(0, 0, 1))
        {
        }

    //! Construct a rotmat3 from an axis and an angle.
    /*! \param axis angle to represent
        \param theta angle to represent

        This is a convenience function for easy initialization of rotmat3s from an axis and an
       angle. The rotmat3 will initialize to the same rotation as the angle around the specified
       axis.
    */
    DEVICE static rotmat3 fromAxisAngle(const vec3<Real>& axis, const Real& theta)
        {
        return rotmat3<Real>(quat<Real>::fromAxisAngle(axis, theta));
        }

    vec3<Real> row0; //!< First row
    vec3<Real> row1; //!< Second row
    vec3<Real> row2; //!< Third row
    };

//! Matrix vector multiplication
/*! \param A matrix
    \param b vector
    \returns A*b

    Multiplication is matrix multiplication, where the vector is represented as a column vector.
*/
template<class Real> DEVICE inline vec3<Real> operator*(const rotmat3<Real>& A, const vec3<Real>& b)
    {
    return vec3<Real>(dot(A.row0, b), dot(A.row1, b), dot(A.row2, b));
    }

//! Matrix matrix multiplication
/*! \param A matrix
    \param B matrix
    \returns A*b
*/
template<class Real>
DEVICE inline rotmat3<Real> operator*(const rotmat3<Real>& A, const rotmat3<Real>& B)
    {
    rotmat3<Real> r;
    rotmat3<Real> B_t = transpose(B);
    r.row0.x = dot(A.row0, B_t.row0);
    r.row0.y = dot(A.row0, B_t.row1);
    r.row0.z = dot(A.row0, B_t.row2);
    r.row1.x = dot(A.row1, B_t.row0);
    r.row1.y = dot(A.row1, B_t.row1);
    r.row1.z = dot(A.row1, B_t.row2);
    r.row2.x = dot(A.row2, B_t.row0);
    r.row2.y = dot(A.row2, B_t.row1);
    r.row2.z = dot(A.row2, B_t.row2);
    return r;
    }

//! Transpose a rotmat3
/*! \param A matrix
    \returns the transpose of A

    A rotation matrix has an inverse equal to its transpose. There may be times where an algorithm
   needs to undo a rotation, so the transpose method is provided.
*/
template<class Real> DEVICE inline rotmat3<Real> transpose(const rotmat3<Real>& A)
    {
    return rotmat3<Real>(vec3<Real>(A.row0.x, A.row1.x, A.row2.x),
                         vec3<Real>(A.row0.y, A.row1.y, A.row2.y),
                         vec3<Real>(A.row0.z, A.row1.z, A.row2.z));
    }

/////////////////////////////// generic operations /////////////////////////////////

//! Vector projection
/*! \param a first vector
    \param b second vector
    \returns the projection of a onto b
    \note projection() can be applied to 2d or 3d vectors
*/
template<class Vec> DEVICE inline Vec project(const Vec& a, const Vec& b)
    {
    return dot(a, b) / dot(b, b) * b;
    }

    } // namespace fresnel

#undef DEVICE

#endif
