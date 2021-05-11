// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_CONVEX_POLYHEDRON_H_
#define GEOMETRY_CONVEX_POLYHEDRON_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Geometry.h"

namespace fresnel
    {
namespace cpu
    {
//! Convex polyhedron geometry
/*! Define a convex polyhedron geometry.

    Attempt 1 at supporting convex polyhedron geometries. Store convex a convex polyhedron as a list
   of planes and perform the needed ray-plane intersection tests to find intersections. Determine
   edge distance by computing plane-plane intersections to find edges. This method works, it is
   unclear if it performs well compared to possible other methods.
*/
class GeometryConvexPolyhedron : public Geometry
    {
    public:
    //! Constructor
    GeometryConvexPolyhedron(
        std::shared_ptr<Scene> scene,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>
            plane_origins,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>
            plane_normals,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>
            plane_colors,
        unsigned int N,
        float r);
    //! Destructor
    virtual ~GeometryConvexPolyhedron();

    //! Get the position buffer
    std::shared_ptr<Array<vec3<float>>> getPositionBuffer()
        {
        return m_position;
        }

    //! Get the orientation buffer
    std::shared_ptr<Array<quat<float>>> getOrientationBuffer()
        {
        return m_orientation;
        }

    //! Get the color buffer
    std::shared_ptr<Array<RGB<float>>> getColorBuffer()
        {
        return m_color;
        }

    //! Set the color by face option
    void setColorByFace(float f)
        {
        m_color_by_face = f;
        }

    //! Get the color by face option
    float getColorByFace() const
        {
        return m_color_by_face;
        }

    protected:
    std::vector<vec3<float>> m_plane_origin; //!< Origins of all the planes in the convex polyhedron
    std::vector<vec3<float>> m_plane_normal; //!< Normals of all the planes in the convex polyhedron
    std::vector<RGB<float>> m_plane_color; //!< Colors assigned to the polyhedron planes

    std::shared_ptr<Array<vec3<float>>> m_position; //!< Position of each polyhedron
    std::shared_ptr<Array<quat<float>>> m_orientation; //!< Orientation of each polyhedron
    std::shared_ptr<Array<RGB<float>>> m_color; //!< Per-particle color

    float m_radius = 0; //!< Precomputed radius
    float m_color_by_face = 0.0f; //!< Flag that mixes per particle color with per face color

    //! Embree bounding function
    static void bounds(const struct RTCBoundsFunctionArguments* args);

    //! Embree ray intersection function
    static void intersect(const struct RTCIntersectFunctionNArguments* args);
    };

//! Export GeometryConvexPolyhedron to python
void export_GeometryConvexPolyhedron(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
