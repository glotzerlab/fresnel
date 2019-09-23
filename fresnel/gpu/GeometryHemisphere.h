// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_HEMISPHERE_H_
#define GEOMETRY_HEMISPHERE_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace gpu {



//! Hemisphere geometry
/*! Define a hemisphere geometry.

    See fresnel::cpu::GeometryHemisphere for full API and description. This class re-implements that using OptiX.
*/

class GeometryHemisphere : public Geometry
    {
    public:
        //! Constructor
        GeometryHemisphere(std::shared_ptr<Scene> scene, unsigned int N);

        //! Destructor
        virtual ~GeometryHemisphere();

        //! Get the position buffer
        std::shared_ptr< Array< vec3<float> > > getPositionBuffer()
            {
            return m_position;
            }

        //! Get the orientation buffer
        std::shared_ptr< Array< quat<float> > > getOrientationBuffer()
            {
            return m_orientation;
            }

        //! Get the radius buffer
        std::shared_ptr< Array< float > > getRadiusBuffer()
            {
            return m_radius;
            }

        //! Get the director buffer
        std::shared_ptr< Array< vec3<float> > > getDirectorBuffer()
            {
            return m_director;
            }

        //! Get the color buffer
        std::shared_ptr< Array< RGB<float> > > getColorBuffer()
            {
            return m_color;
            }

    protected:
        std::shared_ptr< Array< vec3<float> > > m_position;  //!< Position for each hemisphere
        std::shared_ptr< Array< quat<float> > > m_orientation;//!< Orientation for each hemisphere
        std::shared_ptr< Array< float> > m_radius;           //!< Per-particle radii
        std::shared_ptr< Array< vec3<float> > > m_director;  //!< Director for each hemisphere
        std::shared_ptr< Array< RGB<float> > > m_color;      //!< Per-particle color
    };

//! Export GeometryHemisphere to python
void export_GeometryHemisphere(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
