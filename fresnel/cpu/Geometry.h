// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "embree_platform.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <pybind11/pybind11.h>

#include "Scene.h"
#include "common/Material.h"

namespace fresnel { namespace cpu {

//! Handle basic geometry methods
/*! Geometry tracks Embree Geometry objects that belong to a given Scene. Because of Embree's API, the memory management
    for these is handled in a somewhat strange way. A Geometry object adds itself to the Scene on construction. When
    destructed, the Geometry is removed from the Scene. This requires that the user hold onto the Geometry shared pointer
    as long as they want to keep the object live in the scene. There is also an explicit remove function. When a given
    Geometry object is removed from a Scene, calls to change that geometry will fail.

    The base class Geometry itself does not define geometry. It just provides common methods and memory management.
    For derived classes, the bool value m_valid is true when the Geometry is added to the scene. Derived classes
    should set m_valid to true after they successfully call rtcNewWhaetever. m_rtc_geometry stores the geometry id
    returned by Embree to reference this geometry in the scene.

    Each Geometry has a Material and an outline Material and an outline width, but these are managed by Scene. Scene
    has to manage these data structures because of the callback structure of embree ray tracing. In the main tracing
    kernel, only the scene and geometry id are available.
*/
class Geometry
    {
    public:
        //! Constructor
        Geometry(std::shared_ptr<Scene> scene);
        //! Destructor
        virtual ~Geometry();

        //! Enable the Geometry
        void enable();

        //! Disable the Geometry
        void disable();

        //! Remove the Geometry from the Scene
        void remove();

        //! Get the material
        const Material& getMaterial()
            {
            return m_scene->getMaterial(m_geom_id);
            }

        //! Get the outline material
        const Material& getOutlineMaterial()
            {
            return m_scene->getOutlineMaterial(m_geom_id);
            }

        //! Get the outline width
        float getOutlineWidth()
            {
            return m_scene->getOutlineWidth(m_geom_id);
            }

        //! Set the material
        void setMaterial(const Material& material)
            {
            m_scene->setMaterial(m_geom_id, material);
            }

        //! Set the outline material
        void setOutlineMaterial(const Material& material)
            {
            m_scene->setOutlineMaterial(m_geom_id, material);
            }

        //! Set the outline width
        void setOutlineWidth(float width)
            {
            m_scene->setOutlineWidth(m_geom_id, width);
            }

        //! Notify the geometry that changes have been made to the buffers
        void update()
            {
            rtcCommitGeometry(rtcGetGeometry(m_scene->getRTCScene(),m_geom_id));
            }
    protected:
        unsigned int m_geom_id;            //!< Associated geometry id
        bool m_valid=false;                //!< true when the geometry is valid and attached to the Scene
        std::shared_ptr<Scene> m_scene;    //!< The scene the geometry is attached to
        std::shared_ptr<Device> m_device;  //!< The device the Scene is attached to
    };

//! Export Geometry to python
void export_Geometry(pybind11::module& m);

} } // end namespace fresnel::cpu

#endif
