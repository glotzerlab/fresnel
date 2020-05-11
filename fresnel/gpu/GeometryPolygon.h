// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_POLYGON_H_
#define GEOMETRY_POLYGON_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
//! Polygon geometry
/*! Define a polygon geometry.

    See fresnel::gpu::GeometryPolygon for full API and description. This class re-implements that
   using OptiX.
*/
class GeometryPolygon : public Geometry
    {
    public:
    //! Constructor
    GeometryPolygon(
        std::shared_ptr<Scene> scene,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
        float rounding_radius,
        unsigned int N);
    //! Destructor
    virtual ~GeometryPolygon();

    //! Get the position buffer
    std::shared_ptr<Array<vec2<float>>> getPositionBuffer()
        {
        return m_position;
        }

    //! Get the angle buffer
    std::shared_ptr<Array<float>> getAngleBuffer()
        {
        return m_angle;
        }

    //! Get the color buffer
    std::shared_ptr<Array<RGB<float>>> getColorBuffer()
        {
        return m_color;
        }

    //! Get the radius
    float getRadius()
        {
        return m_radius;
        }

    protected:
    optix::Buffer m_vertices; //!< Buffer containing polygon vertices

    std::shared_ptr<Array<vec2<float>>> m_position; //!< Position of each polygon
    std::shared_ptr<Array<float>> m_angle;          //!< Orientation of each polygon
    std::shared_ptr<Array<RGB<float>>> m_color;     //!< Per-particle color

    float m_radius = 0; //!< Precomputed radius in the xy plane
    };

//! Export GeometryPolygon to python
void export_GeometryPolygon(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
