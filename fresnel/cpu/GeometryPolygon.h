// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOMETRY_POLYGON_H_
#define GEOMETRY_POLYGON_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace cpu
    {
//! Polygon geometry
/*! Define a polygon geometry.

    It supports simple polygons in the x,y,z=0 plane.
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
    std::vector<vec2<float>> m_vertices; //!< Polygon vertices
    float m_rounding_radius; //!< Spheropolygon rounding radius

    std::shared_ptr<Array<vec2<float>>> m_position; //!< Position of each polygon
    std::shared_ptr<Array<float>> m_angle; //!< Orientation of each polygon
    std::shared_ptr<Array<RGB<float>>> m_color; //!< Per-particle color

    float m_radius = 0; //!< Precomputed radius in the xy plane

    //! Embree bounding function
    static void bounds(const struct RTCBoundsFunctionArguments* args);

    //! Embree ray intersection function
    static void intersect(const struct RTCIntersectFunctionNArguments* args);
    };

//! Export GeometryPolygon to python
void export_GeometryPolygon(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
