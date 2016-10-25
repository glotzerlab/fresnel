// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_PRISM_H_
#define GEOMETRY_PRISM_H_

#include "embree_platform.h"
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Geometry.h"

namespace fresnel { namespace cpu {

//! Prism geometry
/*! Define a prism geometry.

    Specifically, this class is currently limited to right convex prisms oriented such that the polygon face is
    parallel to the xy plane.

    The initial implementation takes in std::vector's of tuples which will be translated from python
    lists of tuples. This API will be replaced by direct data buffers at some point.

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
                      const std::vector<std::tuple<float, float> > &vertices,
                      const std::vector<std::tuple<float, float> > &position,
                      const std::vector<float> &orientation,
                      const std::vector<float> &height,
                      const std::vector<std::tuple<float, float, float> > &color);
        //! Destructor
        virtual ~GeometryPrism();

    protected:
        std::vector< vec3<float> > m_plane_origin;  //!< Origins of all the planes in the convex polyhedron
        std::vector< vec3<float> > m_plane_normal;  //!< Normals of all the planes in the convex polyhedron

        std::vector< vec3<float> > m_position;      //!< Position of each polyhedron
        std::vector< quat<float> > m_orientation;   //!< Orientation of each polyhedron
        std::vector< float > m_height;              //!< Height of each prism

        std::vector< RGB<float> > m_color;          //!< Per particle color

        float m_radius=0;                           //!< Precomputed radius in the xy plane

        //! Embree bounding function
        static void bounds(void *ptr, size_t item, RTCBounds& bounds_o);

        //! Embree ray intersection function
        static void intersect(void *ptr, RTCRay& ray, size_t item);

        //! Embree ray occlusion function
        static void occlude(void *ptr, RTCRay& ray, size_t item);
    };

//! Export GeometryPrism to python
void export_GeometryPrism(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
