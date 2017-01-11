// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef ARRAY_H_
#define ARRAY_H_

#include "common/ColorMath.h"
#include "common/VectorMath.h"

#include <pybind11/pybind11.h>

// setup pybind11 to use std::shared_ptr
PYBIND11_DECLARE_HOLDER_TYPE(T_shared_ptr_bind, std::shared_ptr<T_shared_ptr_bind>);

namespace fresnel { namespace cpu {

namespace detail
    {
    template <class T>
    unsigned int array_width(const T& a)
        {
        return 1;
        }

    template <class T>
    std::string array_dtype(const T& a)
        {
        return pybind11::format_descriptor<T>::value;
        }

    template <class T>
    unsigned int array_width(const vec2<T>& a)
        {
        return 2;
        }

    template <class T>
    std::string array_dtype(const vec2<T>& a)
        {
        return pybind11::format_descriptor<T>::value;
        }

    template <class T>
    unsigned int array_width(const vec3<T>& a)
        {
        return 3;
        }

    template <class T>
    std::string array_dtype(const vec3<T>& a)
        {
        return pybind11::format_descriptor<T>::value;
        }

    template <class T>
    unsigned int array_width(const quat<T>& a)
        {
        return 4;
        }

    template <class T>
    std::string array_dtype(const quat<T>& a)
        {
        return pybind11::format_descriptor<T>::value;
        }

    template <class T>
    unsigned int array_width(const RGBA<T>& a)
        {
        return 4;
        }

    template <class T>
    std::string array_dtype(const RGBA<T>& a)
        {
        return pybind11::format_descriptor<T>::value;
        }


    template <class T>
    unsigned int array_width(const RGB<T>& a)
        {
        return 3;
        }

    template <class T>
    std::string array_dtype(const RGB<T>& a)
        {
        return pybind11::format_descriptor<T>::value;
        }
    }

//! Array data
/*! Define an array data structure

    This array class encapsulates a 1 or 2 dimensional array of data elements. It is designed to be an abstract
    way of accessing and storing data that is available to Embree and OptiX from python. It works with a proxy class
    written in python to allow map/unmap semantics implicitly while the user sees a numpy-like interface.

    Array should be used for python-facing data structures that the user modifies directly. Internal storage buffers
    should be kept in whatever internal storage is appropriate - use of the Array class signifies that this will be
    directly user-accessible.

    Arrays of vector types (vec3, RGBA, etc...) automatically map to WxHx3 (or 4) numpy arrays to allow users natural
    access to the individual data elements from within python.
*/
template<class T> class Array
    {
    public:
        //! Default constructor
        Array()
            {
            m_w = m_h = 0;
            m_ndim = 1;
            }

        //! Construct a 1D array
        Array(size_t n) : m_data(n)
            {
            m_w = n;
            m_h = 1;
            m_ndim = 1;
            }

        //! Construct a 2D array
        Array(size_t w, size_t h) : m_data(w*h)
            {
            m_w = w;
            m_h = h;
            m_ndim = 2;
            }

        //! Get a python buffer pointing to the data
        pybind11::buffer_info getBuffer()
            {
            std::vector<size_t> shape;
            std::vector<size_t> strides;

            unsigned int array_width = detail::array_width(T());
            size_t item_size = sizeof(T) / array_width;

            //! build up the shape and strides arrays
            if (m_ndim == 1)
                {
                if (array_width == 1)
                    {
                    shape = { m_w };
                    strides = { item_size };
                    }
                else
                    {
                    shape = { m_w, array_width };
                    strides = { item_size*array_width, item_size };
                    }
                }
            else
                {
                if (array_width == 1)
                    {
                    shape = { m_h, m_w };
                    strides = { m_w * item_size, item_size };
                    }
                else
                    {
                    shape = { m_h, m_w, array_width };
                    strides = { m_w*item_size*array_width, item_size*array_width, item_size };
                    }
                }

            unsigned int dim = m_ndim;
            if (array_width > 1)
                dim += 1;

            return pybind11::buffer_info(&m_data[0],
                                         item_size,
                                         detail::array_dtype(T()),
                                         dim,
                                         shape,
                                         strides);
            }

        //! Get the width of the array
        size_t getW()
            {
            return m_w;
            }

        //! Get the height of the array
        size_t getH()
            {
            return m_h;
            }

        //! Get the number of dimensions in the array
        unsigned int getNDim()
            {
            return m_ndim;
            }

        //! Data accessor
        const T& get(size_t i) const
            {
            return m_data[i];
            }

        //! Bind the array
        T* map()
            {
            return &m_data[0];
            }

        //! Map from python
        void map_py()
            {
            // it is important that the python mapping method does not return a pointer
            // if you return a bare pointer with pybind11, pybind11 will try to free it!
            }

        //! Unbind the array
        void unmap()
            {
            }

    protected:
        std::vector<T> m_data;  //!< Stored data
        size_t m_w;             //!< Width of data array
        size_t m_h;             //!< Height of data array
        unsigned int m_ndim;    //!< Number of dimensions in the data array
    };

//! Export Array instantiations to python
void export_Array(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
