// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef ARRAY_H_
#define ARRAY_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "common/ColorMath.h"
#include "common/VectorMath.h"

#if (PYBIND11_VERSION_MAJOR) != 2 || (PYBIND11_VERSION_MINOR) < 2
#error Fresnel requires pybind11 >= 2.2
#endif

namespace fresnel
    {
namespace gpu
    {
namespace detail
    {
template<class T> unsigned int array_width(const T& a)
    {
    return 1;
    }

template<class T> std::string array_dtype(const T& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const vec2<T>& a)
    {
    return 2;
    }

template<class T> std::string array_dtype(const vec2<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const vec3<T>& a)
    {
    return 3;
    }

template<class T> std::string array_dtype(const vec3<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const quat<T>& a)
    {
    return 4;
    }

template<class T> std::string array_dtype(const quat<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const RGBA<T>& a)
    {
    return 4;
    }

template<class T> std::string array_dtype(const RGBA<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }

template<class T> unsigned int array_width(const RGB<T>& a)
    {
    return 3;
    }

template<class T> std::string array_dtype(const RGB<T>& a)
    {
    return pybind11::format_descriptor<T>::value;
    }
    } // namespace detail

//! Array data
/*! Define an array data structure

    See fresnel::cpu::Array for API documentation. This class re-implements that one using optix
   buffers as the internal storage.

    The main difference with gpu::Array is that it manages access into an optix::Buffer, which is
   passed in the constructor. For convenience and proper reference counting, the Array takes
   ownership of the buffer and will destroy it when the ref count on the Array goes to 0
   (annoyingly, optix ref counted objects do not self-destroy).
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

    //! Reference an ndim array to a buffer
    Array(unsigned int ndim, optix::Buffer buffer)
        {
        if (ndim == 1)
            {
            buffer->getSize(m_w);
            m_h = 1;
            m_ndim = 1;
            m_buffer = buffer;
            }
        else if (ndim == 2)
            {
            buffer->getSize(m_w, m_h);
            m_ndim = 2;
            m_buffer = buffer;
            }
        else
            {
            throw std::runtime_error("Invalid dimensions in Array");
            }
        }

    ~Array()
        {
        m_buffer->destroy();
        }

    //! Get a python buffer pointing to the data
    pybind11::buffer_info getBuffer()
        {
        if (m_tmp == nullptr)
            throw std::runtime_error("Array must be mapped before it can be accessed");

        std::vector<size_t> shape;
        std::vector<size_t> strides;

        unsigned int array_width = detail::array_width(T());
        size_t item_size = sizeof(T) / array_width;

        //! build up the shape and strides arrays
        if (m_ndim == 1)
            {
            if (array_width == 1)
                {
                shape = {m_w};
                strides = {item_size};
                }
            else
                {
                shape = {m_w, array_width};
                strides = {item_size * array_width, item_size};
                }
            }
        else
            {
            if (array_width == 1)
                {
                shape = {m_h, m_w};
                strides = {m_w * item_size, item_size};
                }
            else
                {
                shape = {m_h, m_w, array_width};
                strides = {m_w * item_size * array_width, item_size * array_width, item_size};
                }
            }

        unsigned int dim = m_ndim;
        if (array_width > 1)
            dim += 1;

        return pybind11::buffer_info(m_tmp,
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

    //! Map from python
    void map_py()
        {
        // it is important that the python mapping method does not return a pointer
        // if you return a bare pointer with pybind11, pybind11 will try to free it!
        m_tmp = m_buffer->map();
        }

    //! Unbind the array
    void unmap()
        {
        m_tmp = nullptr;
        m_buffer->unmap();
        }

    protected:
    size_t m_w; //!< Width of data array
    size_t m_h; //!< Height of data array
    unsigned int m_ndim; //!< Number of dimensions in the data array
    optix::Buffer m_buffer; //!< OptiX buffer
    void* m_tmp = nullptr; //!< Temporary storage of the data pointer
    };

//! Export Array instantiations to python
void export_Array(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
