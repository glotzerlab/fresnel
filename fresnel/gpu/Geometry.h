// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>

#include "Scene.h"
#include "common/Material.h"

namespace fresnel
    {
namespace gpu
    {
//! Create and modify geometry instances
/*! Geometry tracks OptiX GeometryInstance objects that belong to a given Scene.

    This matches the memory management strategy in cpu::Geometry:
    A Geometry object adds itself to the Scene on construction. When
    destructed, the Geometry is removed from the Scene. This requires that the user hold onto the
   Geometry shared pointer as long as they want to keep the object live in the scene. There is also
   an explicit remove function. When a given Geometry object is removed from a Scene, calls to
   change that geometry will fail.

    The base class Geometry itself does not define geometry. It just provides common methods and
   memory management. For derived classes, the bool value m_valid is true when the Geometry is added
   to the scene.

    Note: Do not confuse Geometry with optix::Geometry in the following:
    The full configuration of a Geometry is split between the base class generic methods and the
   derived class specifics. First, the derived class should allocate memory, buffers, set
   intersection programs, and initialize the optix::Geometry member variable m_geometry. Second, the
   derived class should call setupInstance(), which will create the optix::GeometryInstance
   m_instance and associate it with m_geometry and the proper optix materials.

    Each Geometry has one Material and one outline Material.
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
        return m_mat;
        }

    //! Get the material
    const Material& getOutlineMaterial()
        {
        return m_outline_mat;
        }

    //! Get the outline width
    float getOutlineWidth()
        {
        return m_outline_width;
        }

    //! Set the material
    void setMaterial(const Material& material);

    //! Set the outline material
    void setOutlineMaterial(const Material& material);

    //! Set the outline width
    void setOutlineWidth(float width);

    //! Notify the geometry that changes have been made to the buffers
    void update()
        {
        // notify the scene that its acceleration structure needs to be rebuilt
        m_scene->update();
        }

    protected:
    optix::GeometryInstance m_instance; //!< The geometry instance object
    optix::Geometry m_geometry;         //!< The geometry object
    bool m_valid = false;   //!< true when the geometry instance is valid and attached to the Scene
    bool m_enabled = false; //!< true when the geometry instance is part of the Scene

    std::shared_ptr<Scene> m_scene;   //!< The scene the geometry is attached to
    std::shared_ptr<Device> m_device; //!< The device the Scene is attached to

    Material m_mat;         //!< material assigned to this geometry
    Material m_outline_mat; //!< outline material assigned to this geometry
    float m_outline_width;  //!< outline width

    //! Set up m_instance
    void setupInstance();
    };

//! Export Geometry to python
void export_Geometry(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
