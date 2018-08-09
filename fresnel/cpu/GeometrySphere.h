// Copyright (c) 2016-2018 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_SPHERE_H_
#define GEOMETRY_SPHERE_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

#include <pybind11/pybind11.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace cpu {

//! Sphere geometry
/*! Define a sphere geometry.

    After construction, data fields are all 0. Users are expected to fill out data fields before using the geometry.
    At the python level, there are convenience methods to specify data fields at the time of construction.

    GeometrySphere represents N spheres, each with a position, radius, and color.
*/
class GeometrySphere : public Geometry
    {
    public:
        //! Constructor
        GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N);
        //! Destructor
        virtual ~GeometrySphere();

        //! Get the position buffer
        std::shared_ptr< Array< vec3<float> > > getPositionBuffer()
            {
            return m_position;
            }

        //! Get the radius buffer
        std::shared_ptr< Array< float > > getRadiusBuffer()
            {
            return m_radius;
            }

        //! Get the color buffer
        std::shared_ptr< Array< RGB<float> > > getColorBuffer()
            {
            return m_color;
            }

    protected:

        std::shared_ptr< Array< vec3<float> > > m_position;  //!< Position for each sphere
        std::shared_ptr< Array< float> > m_radius;           //!< Per-particle radii
        std::shared_ptr< Array< RGB<float> > > m_color;      //!< Per-particle color

        //! Embree bounding function
        static void bounds(const struct RTCBoundsFunctionArguments *args);

        //! Embree ray intersection function
        static void intersect(const struct RTCIntersectFunctionNArguments *args);
    };

//! Export GeometrySphere to python
void export_GeometrySphere(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
