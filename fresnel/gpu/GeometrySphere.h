// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_SPHERE_H_
#define GEOMETRY_SPHERE_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>

#include "Geometry.h"

namespace fresnel { namespace gpu {



//! Sphere geometry
/*! Define a sphere geometry.
    The initial implementation takes in std::vector's of tuples which will be translated from python
    lists of tuples. This API will be replaced by direct data buffers at some point.
*/

class GeometrySphere : public Geometry
    {
    public:
        //! Constructor
        GeometrySphere(std::shared_ptr<Scene> scene,
                      const std::vector<std::tuple<float, float, float> > &position,
                      const std::vector< float > &radius
                      );

        //! Destructor
        ~GeometrySphere();

    protected:
    	// Will have to change this
        std::vector< vec3<float> > m_position;      //!< Position of each polyhedron

        std::vector< float> m_radius;               //!< The per-particle radii

    };

//! Export GeometrySphere to python
void export_GeometrySphere(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
