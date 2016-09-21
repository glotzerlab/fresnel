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
*/
class GeometryPrism : public Geometry
    {
    public:
        //! Constructor
        GeometryPrism(std::shared_ptr<Scene> scene,
                      const std::vector<std::tuple<float, float> > &vertices,
                      const std::vector<std::tuple<float, float> > &position,
                      const std::vector<float> &orientation,
                      const std::vector<float> &height);
        //! Destructor
        virtual ~GeometryPrism();

    protected:
        std::vector< vec3<float> > m_plane_origin;  //!< Origins of all the planes in the convex polyhedron
        std::vector< vec3<float> > m_plane_normal;  //!< Normals of all the planes in the convex polyhedron

        std::vector< vec3<float> > m_position;      //!< Position of each polyhedron
        std::vector< quat<float> > m_orientation;   //!< Orientation of each polyhedron
        std::vector< float > m_height;              //!< Height of each prism
    };

//! Export GeometryPrism to python
void export_GeometryPrism(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
