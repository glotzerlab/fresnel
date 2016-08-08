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

//! 4 element color vector
/*! \tparam Real Data type of the components

    colorRGBA defines simple 4 element color vector. The components are available publicly as .r .g .b, and .a.
    A number of simple operations are defined to make writing color math code easier. These include basic element-wise
    addition, subtraction, division, and multiplication (and += -= *= /=), and similarly division, and
    multiplication by scalars, and negation.
*/
template <class Real>
struct colorRGBA
    {
    //! Construct a colorRGBA
    /*! \param _r r-component
        \param _g g-component
        \param _b b-component
        \param _a a-component
    */
    DEVICE colorRGBA(const Real& _r, const Real& _g, const Real& _b, const Real& _a) : r(_r), g(_g), b(_b), a(_a)
        {
        }

    //! Default construct a 0 vector
    DEVICE colorRGBA() : r(0), g(0), b(0), a(0)
        {
        }

    Real r; //!< r-component of the vector
    Real g; //!< g-component of the vector
    Real b; //!< b-component of the vector
    Real a; //!< a-component of the vector
    };


//! Addition of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r+b.r, a.g+b.g, a.b+b.b, a.a+b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator+(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return colorRGBA<Real>(a.r + b.r,
                           a.g + b.g,
                           a.b + b.b,
                           a.a + b.a);
    }

//! Subtraction of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r-b.r, a.g-b.g, a.b-b.b, a.a-b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator-(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return colorRGBA<Real>(a.r - b.r,
                           a.g - b.g,
                           a.b - b.b,
                           a.a - b.a);
    }

//! Multiplication of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r*b.r, a.g*b.g, a.b*b.b, a.a*b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator*(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return colorRGBA<Real>(a.r * b.r,
                           a.g * b.g,
                           a.b * b.b,
                           a.a * b.a);
    }

//! Division of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r/b.r, a.g/b.g, a.b/b.b, a.a/b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator/(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return colorRGBA<Real>(a.r / b.r,
                           a.g / b.g,
                           a.b / b.b,
                           a.a / b.a);
    }

//! Negation of a colorRGBA
/*! \param a Vector

    Negation is component wise.
    \returns The vector (-a.x, -a.y, -a.z).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator-(const colorRGBA<Real>& a)
    {
    return colorRGBA<Real>(-a.r,
                           -a.g,
                           -a.b,
                           -a.a);
    }


//! Assignment-addition of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r += b.r, a.g += b.g, a.b += b.b, a.a += b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator +=(colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    a.r += b.r;
    a.g += b.g;
    a.b += b.b;
    a.a += b.a;
    return a;
    }

//! Assignment-subtraction of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r -= b.r, a.g -= b.g, a.b -= b.b, a.a -= b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator -=(colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    a.r -= b.r;
    a.g -= b.g;
    a.b -= b.b;
    a.a -= b.a;
    return a;
    }

//! Assignment-multiplication of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r *= b.r, a.g *= b.g, a.b *= b.b, a.a *- b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator *=(colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    a.r *= b.r;
    a.g *= b.g;
    a.b *= b.b;
    a.a *= b.a;
    return a;
    }

//! Assignment-division of two colorRGBAs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r /= b.r, a.g /= b.g, a.b /= b.b, a.a /= b.a).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator /=(colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    a.r /= b.r;
    a.g /= b.g;
    a.b /= b.b;
    a.a /= b.a;
    return a;
    }

//! Multiplication of a colorRGBA by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b, a.a*b).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator*(const colorRGBA<Real>& a, const Real& b)
    {
    return colorRGBA<Real>(a.r * b,
                           a.g * b,
                           a.b * b,
                           a.a * b);
    }

//! Multiplication of a colorRGBA by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b, a.a*b).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator*(const Real& b, const colorRGBA<Real>& a)
    {
    return colorRGBA<Real>(a.r * b,
                           a.g * b,
                           a.b * b,
                           a.a * b);
    }

//! Division of a colorRGBA by a scalar
/*! \param a vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r/b, a.g/b, a.b/b, a.a/b).
*/
template < class Real >
DEVICE inline colorRGBA<Real> operator/(const colorRGBA<Real>& a, const Real& b)
    {
    Real q = Real(1.0)/b;
    return a * q;
    }

//! Assignment-multiplication of a colorRGBA by a scalar
/*! \param a First vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r *= b, a.g *= b, a.b *= b, a.a *= b).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator *=(colorRGBA<Real>& a, const Real& b)
    {
    a.r *= b;
    a.g *= b;
    a.b *= b;
    a.a *= a;
    return a;
    }

//! Assignment-division of a colorRGBA by a scalar
/*! \param a First vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r /= b, a.g /= b, a.b /= b, a.a /= b).
*/
template < class Real >
DEVICE inline colorRGBA<Real>& operator /=(colorRGBA<Real>& a, const Real& b)
    {
    a.r /= b;
    a.g /= b;
    a.b /= b;
    a.a /= b;
    return a;
    }

//! Equality test of two colorRGBAs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template < class Real >
DEVICE inline bool operator ==(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return (a.r == b.r) && (a.g == b.g) && (a.b == b.b) && (a.a == b.a);
    }

//! Inequality test of two colorRGBAs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are not identically equal, and false if they are
*/
template < class Real >
DEVICE inline bool operator !=(const colorRGBA<Real>& a, const colorRGBA<Real>& b)
    {
    return (a.r != b.r) || (a.g != b.g) || (a.b != b.b) || (a.a != b.a);
    }

#undef DEVICE

#endif
