// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_PRISM_H_
#define GEOMETRY_PRISM_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace cpu {

//! Prism geometry
/*! Define a prism geometry.

    Specifically, this class is currently limited to right convex prisms oriented such that the polygon face is
    parallel to the xy plane.

    Verticies are passed in from python as a numpy array.

    While this class is very specific in the type of geometry it supports, the implementation is actually very general.
    The intersection routine is capable of supporting generic 3D polyhedra. This class could be copied and used to
    implement polyhedra, or one could implement multiple initialization routines and a flag for what type.
    of geometry is supported. Here are the changes needed.

    - In bounds(), report an AABB around the bounding sphere, ignore per particle height
    - In intersect(), do not rewrite plane 0 with the height
    - A new initialization routine to set up the shape planes for a general convex polyhedron, and allow 3D position
      and quaternion orientation.
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
        std::vector< vec3<float> > m_plane_origin;  //!< Origins of all the planes in the convex polyhedron
        std::vector< vec3<float> > m_plane_normal;  //!< Normals of all the planes in the convex polyhedron

        std::shared_ptr< Array< vec2<float> > > m_position;   //!< Position of each polyhedron
        std::shared_ptr< Array< float > > m_angle;            //!< Orientation of each polyhedron
        std::shared_ptr< Array< float > > m_height;           //!< Height of each prism
        std::shared_ptr< Array< RGB<float> > > m_color;       //!< Per-particle color

        float m_radius=0;                           //!< Precomputed radius in the xy plane

        //! Embree bounding function
        static void bounds(const struct RTCBoundsFunctionArguments *args);

        //! Embree ray intersection function
        static void intersect(const struct RTCIntersectFunctionNArguments *args);
    };

//! Export GeometryPrism to python
void export_GeometryPrism(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
