// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_ELLIPSOID_H_
#define GEOMETRY_ELLIPSOID_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
//! Ellipsoid geometry
/*! Define an ellipsoid geometry.

    See fresnel::cpu::GeometryEllipsoid for full API and description. This class re-implements that
   using OptiX.
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
    std::shared_ptr<Array<vec3<float>>> m_radii;       //!< Per-particle radii in x,y,z direction
	std::shared_ptr<Array<quat<float>>> m_orientation; //!< Per-particle orientation
    std::shared_ptr<Array<RGB<float>>> m_color;        //!< Per-particle color
    };

//! Export GeometryEllipsoid to python
void export_GeometryEllipsoid(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
