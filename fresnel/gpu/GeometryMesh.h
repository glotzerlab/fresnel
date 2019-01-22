// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_MESH_H_
#define GEOMETRY_MESH_H_

#include <optixu/optixpp_namespace.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Geometry.h"
#include "Array.h"

namespace fresnel { namespace gpu {

//! Mesh geometry
/*! Define a triangulated mesh geometry.

    See fresnel::cpu::GeometryMesh for full API and description. This class re-implements that using OptiX.
*/
class GeometryMesh : public Geometry
    {
    public:
        //! Constructor
        GeometryMesh(std::shared_ptr<Scene> scene,
                     pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                     unsigned int N);
        //! Destructor
        virtual ~GeometryMesh();

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

    protected:
        optix::Buffer m_vertices;      //!< Buffer containing mesh vertices

        std::shared_ptr< Array< vec3<float> > > m_position;     //!< Position of each polyhedron
        std::shared_ptr< Array< quat<float> > > m_orientation;  //!< Orientation of each polyhedron
        std::shared_ptr< Array< RGB<float> > > m_color;         //!< Per-vertex color
    };

//! Export GeometryMesh to python
void export_GeometryMesh(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
