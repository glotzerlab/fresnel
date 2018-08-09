// Copyright (c) 2016-2018 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_CONVEX_POLYHEDRON_H_
#define GEOMETRY_CONVEX_POLYHEDRON_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace gpu {

//! Prism geometry
/*! Define a convex polyhedron geometry.

    See fresnel::cpu::GeometryConvexPolyhedron for full API and description. This class re-implements that using OptiX.
*/
class GeometryConvexPolyhedron : public Geometry
    {
    public:
        //! Constructor
        GeometryConvexPolyhedron(std::shared_ptr<Scene> scene,
                                 pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_origins,
                                 pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_normals,
                                 pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_colors,
                                 unsigned int N,
                                 float r);
        //! Destructor
        virtual ~GeometryConvexPolyhedron();

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

        //! Get the color buffer
        std::shared_ptr< Array< RGB<float> > > getColorBuffer()
            {
            return m_color;
            }

        //! Set the color by face option
        void setColorByFace(float f)
            {
            m_geometry["convex_polyhedron_color_by_face"]->setFloat(f);
            }

        //! Get the color by face option
        float getColorByFace()
            {
            return m_geometry["convex_polyhedron_color_by_face"]->getFloat();
            }

    protected:
        optix::Buffer m_plane_origin;   //!< Buffer containing plane origins
        optix::Buffer m_plane_normal;   //!< Buffer containing plane normals
        optix::Buffer m_plane_color;    //!< Buffer containing plane colors

        std::shared_ptr< Array< vec3<float> > > m_position;     //!< Position of each polyhedron
        std::shared_ptr< Array< quat<float> > > m_orientation;  //!< Orientation of each polyhedron
        std::shared_ptr< Array< RGB<float> > > m_color;         //!< Per-particle color
    };

//! Export GeometryConvexPolyhedron to python
void export_GeometryConvexPolyhedron(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
