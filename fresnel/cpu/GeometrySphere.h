// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_SPHERE_H_
#define GEOMETRY_SPHERE_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#include <pybind11/pybind11.h>

#include "Geometry.h"

namespace fresnel { namespace cpu {

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
        virtual ~GeometrySphere();

    protected:

        std::vector< vec3<float> > m_position;      //!< Position of each polyhedron

        std::vector< float> m_radius;               //!< The per-particle radii

        //! Embree bounding function
        static void bounds(void *ptr, size_t item, RTCBounds& bounds_o);

        //! Embree ray intersection function
        static void intersect(void *ptr, RTCRay& ray, size_t item);

        //! Embree ray occlusion function
        static void occlude(void *ptr, RTCRay& ray, size_t item);
    };

//! Export GeometrySphere to python
void export_GeometrySphere(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
