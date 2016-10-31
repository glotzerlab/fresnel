// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_PRISM_H_
#define GEOMETRY_PRISM_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Geometry.h"

namespace fresnel { namespace gpu {

//! Prism geometry
/*! Define a prism geometry.

    See fresnel::cpu::GeometryPrism for full API and description. This class re-implements that using OptiX.
*/
class GeometryPrism : public Geometry
    {
    public:
        //! Constructor
        GeometryPrism(std::shared_ptr<Scene> scene,
                      pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                      unsigned int N);
        //! Destructor
        virtual ~GeometryPrism();

        //! Get the position buffer
        std::shared_ptr< Array< vec2<float> > > getPositionBuffer()
            {
            return m_position;
            }

        //! Get the height buffer
        std::shared_ptr< Array< float > > getHeightBuffer()
            {
            return m_height;
            }

        //! Get the angle buffer
        std::shared_ptr< Array< float > > getAngleBuffer()
            {
            return m_angle;
            }

        //! Get the color buffer
        std::shared_ptr< Array< RGB<float> > > getColorBuffer()
            {
            return m_color;
            }

    protected:
        optix::Buffer m_plane_origin;   //!< Buffer containing plane origins
        optix::Buffer m_plane_normal;   //!< Buffer containing plane normals

        std::shared_ptr< Array< vec2<float> > > m_position;   //!< Position of each polyhedron
        std::shared_ptr< Array< float > > m_angle;            //!< Orientation of each polyhedron
        std::shared_ptr< Array< float > > m_height;           //!< Height of each prism
        std::shared_ptr< Array< RGB<float> > > m_color;       //!< Per-particle color
    };

//! Export GeometryPrism to python
void export_GeometryPrism(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
