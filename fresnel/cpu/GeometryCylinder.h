// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_CYLINDER_H_
#define GEOMETRY_CYLINDER_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#include <pybind11/pybind11.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace cpu {

//! Cylinder geometry
/*! Define a cylinder geometry

    After construction, data fields are all 0. Users are expected to fill out data fields before using the geometry.
    At the python level, there are convenience methods to specify data fields at the time of construction.

    GeometryCylinder represents N spherocylinders, each defined by start and end positions, a radius, and a color.
*/
class GeometryCylinder : public Geometry
    {
    public:
        //! Constructor
        GeometryCylinder(std::shared_ptr<Scene> scene, unsigned int N);
        //! Destructor
        virtual ~GeometryCylinder();

        //! Get the A buffer
        std::shared_ptr< Array< vec3<float> > > getABuffer()
            {
            return m_A;
            }

        //! Get the B buffer
        std::shared_ptr< Array< vec3<float> > > getBBuffer()
            {
            return m_B;
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

        std::shared_ptr< Array< vec3<float> > > m_A;         //!< Position the start of each cylinder
        std::shared_ptr< Array< vec3<float> > > m_B;         //!< Position the end of each cylinder
        std::shared_ptr< Array< float> > m_radius;           //!< Per-particle radii
        std::shared_ptr< Array< RGB<float> > > m_color;      //!< Per-particle color

        //! Embree bounding function
        static void bounds(void *ptr, size_t item, RTCBounds& bounds_o);

        //! Embree ray intersection function
        static void intersect(void *ptr, RTCRay& ray, size_t item);
    };

//! Export Cylinder to python
void export_GeometryCylinder(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
