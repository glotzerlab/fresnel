// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __COLOR_MATH_H__
#define __COLOR_MATH_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

//! 3 element color vector
/*! \tparam Real Data type of the components

    RGB defines simple 3 element color vector. The components are available publicly as .r .g .b.
*/
template <class Real>
struct RGB
    {
    //! Construct a RGB
    /*! \param _r r-component
        \param _g g-component
        \param _b b-component
    */
    DEVICE RGB(const Real& _r, const Real& _g, const Real& _b) : r(_r), g(_g), b(_b)
        {
        }

    //! Default construct a 0 vector
    DEVICE RGB() : r(0), g(0), b(0)
        {
        }

    #ifdef NVCC
    //! Convenience function to generate float3 in device code
    DEVICE operator float3()
        {
        return make_float3(r, g, b);
        }

    //! Convenience function to get a vec3 from a float3 in device code
    DEVICE explicit RGB(const float3& a) : r(a.x), g(a.y), b(a.z)
        {
        }
    #endif


    Real r; //!< r-component of the vector
    Real g; //!< g-component of the vector
    Real b; //!< b-component of the vector
    };

//! Addition of two RGBs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r+b.r, a.g+b.g, a.b+b.b).
*/
template < class Real >
DEVICE inline RGB<Real> operator+(const RGB<Real>& a, const RGB<Real>& b)
    {
    return RGB<Real>(a.r + b.r,
                           a.g + b.g,
                           a.b + b.b);
    }

//! Subtraction of two RGBs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r-b.r, a.g-b.g, a.b-b.b).
*/
template < class Real >
DEVICE inline RGB<Real> operator-(const RGB<Real>& a, const RGB<Real>& b)
    {
    return RGB<Real>(a.r - b.r,
                           a.g - b.g,
                           a.b - b.b);
    }

//! Multiplication of two RGBs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r*b.r, a.g*b.g, a.b*b.b).
*/
template < class Real >
DEVICE inline RGB<Real> operator*(const RGB<Real>& a, const RGB<Real>& b)
    {
    return RGB<Real>(a.r * b.r,
                           a.g * b.g,
                           a.b * b.b);
    }

//! Division of two RGBs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r/b.r, a.g/b.g, a.b/b.b).
*/
template < class Real >
DEVICE inline RGB<Real> operator/(const RGB<Real>& a, const RGB<Real>& b)
    {
    return RGB<Real>(a.r / b.r,
                           a.g / b.g,
                           a.b / b.b);
    }

//! Negation of a RGB
/*! \param a Vector

    Negation is component wise.
    \returns The vector (-a.x, -a.y, -a.z).
*/
template < class Real >
DEVICE inline RGB<Real> operator-(const RGB<Real>& a)
    {
    return RGB<Real>(-a.r,
                           -a.g,
                           -a.b);
    }


//! Assignment-addition of two RGBs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r += b.r, a.g += b.g, a.b += b.b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator +=(RGB<Real>& a, const RGB<Real>& b)
    {
    a.r += b.r;
    a.g += b.g;
    a.b += b.b;
    return a;
    }

//! Assignment-subtraction of two RGBs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r -= b.r, a.g -= b.g, a.b -= b.b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator -=(RGB<Real>& a, const RGB<Real>& b)
    {
    a.r -= b.r;
    a.g -= b.g;
    a.b -= b.b;
    return a;
    }

//! Assignment-multiplication of two RGBs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r *= b.r, a.g *= b.g, a.b *= b.b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator *=(RGB<Real>& a, const RGB<Real>& b)
    {
    a.r *= b.r;
    a.g *= b.g;
    a.b *= b.b;
    return a;
    }

//! Assignment-division of two RGBs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r /= b.r, a.g /= b.g, a.b /= b.b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator /=(RGB<Real>& a, const RGB<Real>& b)
    {
    a.r /= b.r;
    a.g /= b.g;
    a.b /= b.b;
    return a;
    }

//! Multiplication of a RGB by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b).
*/
template < class Real >
DEVICE inline RGB<Real> operator*(const RGB<Real>& a, const Real& b)
    {
    return RGB<Real>(a.r * b,
                           a.g * b,
                           a.b * b);
    }

//! Multiplication of a RGB by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b).
*/
template < class Real >
DEVICE inline RGB<Real> operator*(const Real& b, const RGB<Real>& a)
    {
    return RGB<Real>(a.r * b,
                           a.g * b,
                           a.b * b);
    }

//! Division of a RGB by a scalar
/*! \param a vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r/b, a.g/b, a.b/b).
*/
template < class Real >
DEVICE inline RGB<Real> operator/(const RGB<Real>& a, const Real& b)
    {
    Real q = Real(1.0)/b;
    return a * q;
    }

//! Assignment-multiplication of a RGB by a scalar
/*! \param a First vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r *= b, a.g *= b, a.b *= b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator *=(RGB<Real>& a, const Real& b)
    {
    a.r *= b;
    a.g *= b;
    a.b *= b;
    return a;
    }

//! Assignment-division of a RGB by a scalar
/*! \param a First vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r /= b, a.g /= b, a.b /= b).
*/
template < class Real >
DEVICE inline RGB<Real>& operator /=(RGB<Real>& a, const Real& b)
    {
    a.r /= b;
    a.g /= b;
    a.b /= b;
    return a;
    }

//! Equality test of two RGBs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template < class Real >
DEVICE inline bool operator ==(const RGB<Real>& a, const RGB<Real>& b)
    {
    return (a.r == b.r) && (a.g == b.g) && (a.b == b.b);
    }

//! Inequality test of two RGBs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are not identically equal, and false if they are
*/
template < class Real >
DEVICE inline bool operator !=(const RGB<Real>& a, const RGB<Real>& b)
    {
    return (a.r != b.r) || (a.g != b.g) || (a.b != b.b);
    }

//! 4 element color vector
/*! \tparam Real Data type of the components

    RGBA defines simple 4 element color vector. The components are available publicly as .r .g .b, and .a.
*/
template <class Real>
struct RGBA
    {
    //! Construct a RGBA
    /*! \param _r r-component
        \param _g g-component
        \param _b b-component
        \param _a a-component
    */
    DEVICE RGBA(const Real& _r, const Real& _g, const Real& _b, const Real& _a) : r(_r), g(_g), b(_b), a(_a)
        {
        }

    //! Construct a RGBA from a RGB
    /*! \param c Color to provide r,g,b components.
        \param _a a-component
    */
    DEVICE RGBA(const RGB<Real>& c, const Real& _a) : r(c.r), g(c.g), b(c.b), a(a)
        {
        }


    //! Default construct a 0 vector
    DEVICE RGBA() : r(0), g(0), b(0), a(1.0)
        {
        }

    Real r; //!< r-component of the vector
    Real g; //!< g-component of the vector
    Real b; //!< b-component of the vector
    Real a; //!< a-component of the vector
    };

//! Linear interpolate between two values
/*! \param x interpolation fraction
    \param a left side of the interpolation (x=0)
    \param b right side of the interpolation (x=1)

    \returns (1-x)*a + x*b
*/
template <class T>
DEVICE inline T lerp(float x, const T& a, const T& b)
    {
    return (1-x) * a + x * b;
    }

#undef DEVICE

#endif
