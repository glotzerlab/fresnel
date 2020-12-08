// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_SPHERE_H_
#define GEOMETRY_SPHERE_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>

#include "Array.h"
#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
//! Sphere geometry
/*! Define a sphere geometry.

    See fresnel::cpu::GeometrySphere for full API and description. This class re-implements that
   using OptiX.
*/

class GeometrySphere : public Geometry
    {
    public:
    //! Constructor
    GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N);

    //! Destructor
    virtual ~GeometrySphere();

    //! Get the position buffer
    std::shared_ptr<Array<vec3<float>>> getPositionBuffer()
        {
        return m_position;
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
    std::shared_ptr<Array<vec3<float>>> m_position; //!< Position for each sphere
    std::shared_ptr<Array<float>> m_radius;         //!< Per-particle radii
    std::shared_ptr<Array<RGB<float>>> m_color;     //!< Per-particle color
    };

//! Export GeometrySphere to python
void export_GeometrySphere(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
