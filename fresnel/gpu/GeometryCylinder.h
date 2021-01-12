// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_CYLINDER_H_
#define GEOMETRY_CYLINDER_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
//! Cylinder geometry
/*! Define a cylinder geometry

    See fresnel::cpu::GeometryCylinder for full API and description. This class re-implements that
   using OptiX.
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
    std::shared_ptr<Array<float>> m_radius;       //!< Per-particle radii
    std::shared_ptr<Array<RGB<float>>> m_color;   //!< Color for each start and end point
    };

//! Export GeometryCylinder to python
void export_GeometryCylinder(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
