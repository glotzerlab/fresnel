// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_ELLIPSOID_H_
#define GEOMETRY_ELLIPSOID_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace cpu
    {
//! Ellipsoid geometry
/*! Define an ellipsoid geometry.

    After construction, data fields are all 0. Users are expected to fill out data fields before
   using the geometry. At the python level, there are convenience methods to specify data fields at
   the time of construction.

    GeometryEllipsoid represents N ellipsoids, each with a position, major axes, orientation, and color.
*/
class GeometryEllipsoid : public Geometry
    {
    public:
    //! Constructor
    GeometryEllipsoid(std::shared_ptr<Scene> scene, unsigned int N);
    //! Destructor
    virtual ~GeometryEllipsoid();

    //! Get the position buffer
    std::shared_ptr<Array<vec3<float>>> getPositionBuffer()
        {
        return m_position;
        }

    //! Get the radii buffer
    std::shared_ptr<Array<vec3<float>>> getRadiiBuffer()
        {
		return m_radii;
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

    protected:
    std::shared_ptr<Array<vec3<float>>> m_position;    //!< Position for each ellipsoid
    std::shared_ptr<Array<float>> m_radii;             //!< Per-particle radii in x,y,z direction
	std::shared_ptr<Array<quat<float>>> m_orientation; //!< Per-particle orientation
    std::shared_ptr<Array<RGB<float>>> m_color;        //!< Per-particle color

    //! Embree bounding function
    static void bounds(const struct RTCBoundsFunctionArguments* args);

    //! Embree ray intersection function
    static void intersect(const struct RTCIntersectFunctionNArguments* args);
    };

//! Export GeometryEllipsoid to python
void export_GeometryEllipsoid(pybind11::module& m);

    } // namespace cpu
    } // namespace fresnel

#endif
