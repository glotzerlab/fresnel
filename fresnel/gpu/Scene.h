// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef SCENE_H_
#define SCENE_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Device.h"

namespace fresnel { namespace gpu {

//! Manage the root scene graph object
/*! OptiX does not define a concrete notion of a scene. We create one in Fresnel by managing the root object of the
    scene graph as a Scene, for compatibility with the cpu::Scene API. The root object is a Geometry Group which can
    hold any number of geometry instances.

    The Scene also manages an acceleration structure for all of the primitives. Whenever a child object is modified,
    added, or removed, they must mark the acceleration structure dirty.
*/
class Scene
    {
    public:
        //! Constructor
        Scene(std::shared_ptr<Device> device);
        //! Destructor
        ~Scene();

        //! Access the root object
        optix::GeometryGroup& getRoot()
            {
            return m_root;
            }

        //! Access the acceleration structure
        optix::Acceleration& getAccel()
            {
            return m_accel;
            }

        //! Access the Device
        std::shared_ptr<Device> getDevice()
            {
            return m_device;
            }

        //! Add a geometry instance to the scene
        void addGeometry(optix::GeometryInstance inst)
            {
            m_root->addChild(inst);
            m_accel->markDirty();
            }

        //! Remove a geometry instance from the scene
        void removeGeometry(optix::GeometryInstance inst)
            {
            m_root->removeChild(inst);
            m_accel->markDirty();
            }

        //! Update acceleration structures
        /*! Call when any geometry in this scene is modified
        */
        void update()
            {
            m_accel->markDirty();
            }

    private:
        optix::GeometryGroup m_root;        //!< Store the scene root object
        optix::Acceleration m_accel;        //!< Store the acceleration structure
        std::shared_ptr<Device> m_device;   //!< The device the scene is attached to
    };

//! Export Scene to python
void export_Scene(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
