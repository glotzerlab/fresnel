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

    colorRGB defines simple 3 element color vector. The components are available publicly as .r .g .b.
*/
template <class Real>
struct colorRGB
    {
    //! Construct a colorRGB
    /*! \param _r r-component
        \param _g g-component
        \param _b b-component
    */
    DEVICE colorRGB(const Real& _r, const Real& _g, const Real& _b) : r(_r), g(_g), b(_b)
        {
        }

    //! Default construct a 0 vector
    DEVICE colorRGB() : r(0), g(0), b(0)
        {
        }

    Real r; //!< r-component of the vector
    Real g; //!< g-component of the vector
    Real b; //!< b-component of the vector
    };

//! Addition of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r+b.r, a.g+b.g, a.b+b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator+(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return colorRGB<Real>(a.r + b.r,
                           a.g + b.g,
                           a.b + b.b);
    }

//! Subtraction of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r-b.r, a.g-b.g, a.b-b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator-(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return colorRGB<Real>(a.r - b.r,
                           a.g - b.g,
                           a.b - b.b);
    }

//! Multiplication of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r*b.r, a.g*b.g, a.b*b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator*(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return colorRGB<Real>(a.r * b.r,
                           a.g * b.g,
                           a.b * b.b);
    }

//! Division of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r/b.r, a.g/b.g, a.b/b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator/(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return colorRGB<Real>(a.r / b.r,
                           a.g / b.g,
                           a.b / b.b);
    }

//! Negation of a colorRGB
/*! \param a Vector

    Negation is component wise.
    \returns The vector (-a.x, -a.y, -a.z).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator-(const colorRGB<Real>& a)
    {
    return colorRGB<Real>(-a.r,
                           -a.g,
                           -a.b);
    }


//! Assignment-addition of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.r += b.r, a.g += b.g, a.b += b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator +=(colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    a.r += b.r;
    a.g += b.g;
    a.b += b.b;
    return a;
    }

//! Assignment-subtraction of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.r -= b.r, a.g -= b.g, a.b -= b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator -=(colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    a.r -= b.r;
    a.g -= b.g;
    a.b -= b.b;
    return a;
    }

//! Assignment-multiplication of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Multiplication is component wise.
    \returns The vector (a.r *= b.r, a.g *= b.g, a.b *= b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator *=(colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    a.r *= b.r;
    a.g *= b.g;
    a.b *= b.b;
    return a;
    }

//! Assignment-division of two colorRGBs
/*! \param a First vector
    \param b Second vector

    Division is component wise.
    \returns The vector (a.r /= b.r, a.g /= b.g, a.b /= b.b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator /=(colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    a.r /= b.r;
    a.g /= b.g;
    a.b /= b.b;
    return a;
    }

//! Multiplication of a colorRGB by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator*(const colorRGB<Real>& a, const Real& b)
    {
    return colorRGB<Real>(a.r * b,
                           a.g * b,
                           a.b * b);
    }

//! Multiplication of a colorRGB by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r*b, a.g*b, a.b*b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator*(const Real& b, const colorRGB<Real>& a)
    {
    return colorRGB<Real>(a.r * b,
                           a.g * b,
                           a.b * b);
    }

//! Division of a colorRGB by a scalar
/*! \param a vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r/b, a.g/b, a.b/b).
*/
template < class Real >
DEVICE inline colorRGB<Real> operator/(const colorRGB<Real>& a, const Real& b)
    {
    Real q = Real(1.0)/b;
    return a * q;
    }

//! Assignment-multiplication of a colorRGB by a scalar
/*! \param a First vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.r *= b, a.g *= b, a.b *= b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator *=(colorRGB<Real>& a, const Real& b)
    {
    a.r *= b;
    a.g *= b;
    a.b *= b;
    return a;
    }

//! Assignment-division of a colorRGB by a scalar
/*! \param a First vector
    \param b scalar

    Division is component wise.
    \returns The vector (a.r /= b, a.g /= b, a.b /= b).
*/
template < class Real >
DEVICE inline colorRGB<Real>& operator /=(colorRGB<Real>& a, const Real& b)
    {
    a.r /= b;
    a.g /= b;
    a.b /= b;
    return a;
    }

//! Equality test of two colorRGBs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template < class Real >
DEVICE inline bool operator ==(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return (a.r == b.r) && (a.g == b.g) && (a.b == b.b);
    }

//! Inequality test of two colorRGBs
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are not identically equal, and false if they are
*/
template < class Real >
DEVICE inline bool operator !=(const colorRGB<Real>& a, const colorRGB<Real>& b)
    {
    return (a.r != b.r) || (a.g != b.g) || (a.b != b.b);
    }

//! 4 element color vector
/*! \tparam Real Data type of the components

    colorRGBA defines simple 4 element color vector. The components are available publicly as .r .g .b, and .a.
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

    //! Construct a colorRGBA from a colorRGB
    /*! \param c Color to provide r,g,b components.
        \param _a a-component
    */
    DEVICE colorRGBA(const colorRGB<Real>& c, const Real& _a) : r(c.r), g(c.g), b(c.b), a(a)
        {
        }


    //! Default construct a 0 vector
    DEVICE colorRGBA() : r(0), g(0), b(0), a(1.0)
        {
        }

    Real r; //!< r-component of the vector
    Real g; //!< g-component of the vector
    Real b; //!< b-component of the vector
    Real a; //!< a-component of the vector
    };


#undef DEVICE

#endif
