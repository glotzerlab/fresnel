// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef GEOMETRY_CYLINDER_H_
#define GEOMETRY_CYLINDER_H_

#include "embree_platform.h"
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace cpu
    {
//! Cylinder geometry
/*! Define a cylinder geometry

    After construction, data fields are all 0. Users are expected to fill out data fields before
   using the geometry. At the python level, there are convenience methods to specify data fields at
   the time of construction.

    GeometryCylinder represents N spherocylinders, each defined by start and end positions, a
   radius, and a color.

    The start and end points (and colors) are stored in Nx2 arrays.
*/
class GeometryCylinder : public Geometry
    {
    public:
    //! Constructor
    GeometryCylinder(std::shared_ptr<Scene> scene, unsigned int N);
    //! Destructor
    virtual ~GeometryCylinder();

    //! Get the end points buffer
    std::shared_ptr<Array<vec3<float>>> getPointsBuffer()
        {
        return m_points;
        }

    //! Get the radius buffer
    std::shared_ptr<Array<float>> getRadiusBuffer()
        {
        return m_radius;
        }

    //! Get the color buffer
    std::shared_ptr<Array<RGB<float>>> getColorBuffer()
        {
        return m_color;
        }

    protected:
    std::shared_ptr<Array<vec3<float>>> m_points; //!< Position the start and end of each cylinder
    std::shared_ptr<Array<float>> m_radius; //!< Per-particle radii
    std::shared_ptr<Array<RGB<float>>> m_color; //!< Color for each start and end point

    //! Embree bounding function
    static void bounds(const struct RTCBoundsFunctionArguments* args);

    //! Embree ray intersection function
    static void intersect(const struct RTCIntersectFunctionNArguments* args);
    };

//! Export Cylinder to python
void export_GeometryCylinder(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
