// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Geometry.h"

namespace fresnel
    {
namespace gpu
    {
/*! \param scene Scene to attach the Geometry to
    The base class constructor does nothing. It is up to the derived classes to implement
   appropriate geometry creation routines and set m_valid and m_geom_id appropriately.
*/
Geometry::Geometry(std::shared_ptr<Scene> scene) : m_scene(scene), m_device(scene->getDevice()) { }

Geometry::~Geometry()
    {
    remove();
    }

/*! When enabled, the geometry will be present when rendering the scene
 */
void Geometry::enable()
    {
    if (m_valid)
        {
        if (!m_enabled)
            {
            m_scene->addGeometry(m_instance);
            m_enabled = true;
            }
        }
    else
        {
        throw std::runtime_error("Cannot enable inactive Geometry");
        }
    }

/*! When disabled, the geometry will not be present in the scene. No rays will intersect it.
 */
void Geometry::disable()
    {
    if (m_valid)
        {
        if (m_enabled)
            {
            m_enabled = false;
            m_scene->removeGeometry(m_instance);
            }
        }
    else
        {
        throw std::runtime_error("Cannot disable inactive Geometry");
        }
    }

/*! Once it is removed from a Scene, the Geometry cannot be changed, enabled, or disabled.
    remove() may be called multiple times. It has no effect on subsequent calls.
*/
void Geometry::remove()
    {
    if (m_valid)
        {
        m_scene->removeGeometry(m_instance);
        m_instance->destroy();
        m_geometry->destroy();
        m_valid = false;
        m_enabled = false;
        }
    }

/*! Using the optix::Geometry in m_geometry and the material programs, initialize m_instance
 */
void Geometry::setupInstance()
    {
    m_instance = m_device->getContext()->createGeometryInstance();
    m_instance->setGeometry(m_geometry);
    m_instance->setMaterialCount(1);
    m_instance->setMaterial(0, m_device->getMaterial());

    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));

    m_valid = true;
    enable();
    }

/*! \param material New material to set

    Sets the material parameters. These are translated to OptiX variables that are part of the
   geometry intance scope. Also cache a local copy of the material for get methods.
*/
void Geometry::setMaterial(const Material& material)
    {
    m_mat = material;
    m_instance["material"]->setUserData(sizeof(m_mat), &m_mat);
    }

/*! \param material New outline material to set

    Sets the outline material parameters. These are translated to OptiX variables that are part of
   the geometry instance scope. Also cache a local copy of the material for get methods.
*/
void Geometry::setOutlineMaterial(const Material& material)
    {
    m_outline_mat = material;
    m_instance["outline_material"]->setUserData(sizeof(m_outline_mat), &m_outline_mat);
    }

/*! \param material New outline material to set

    Sets the outline material parameters. These are translated to OptiX variables that are part of
   the geometry instance scope. Also cache a local copy of the material for get methods.
*/
void Geometry::setOutlineWidth(float width)
    {
    m_outline_width = width;
    m_instance["outline_width"]->setFloat(width);
    }

/*! \param m Python module to export in
 */
void export_Geometry(pybind11::module& m)
    {
    pybind11::class_<Geometry, std::shared_ptr<Geometry>>(m, "Geometry")
        .def(pybind11::init<std::shared_ptr<Scene>>())
        .def("getMaterial", &Geometry::getMaterial)
        .def("setMaterial", &Geometry::setMaterial)
        .def("getOutlineMaterial", &Geometry::getOutlineMaterial)
        .def("setOutlineMaterial", &Geometry::setOutlineMaterial)
        .def("getOutlineWidth", &Geometry::getOutlineWidth)
        .def("setOutlineWidth", &Geometry::setOutlineWidth)
        .def("disable", &Geometry::disable)
        .def("enable", &Geometry::enable)
        .def("remove", &Geometry::remove)
        .def("update", &Geometry::update);
    }

    } // namespace gpu
    } // namespace fresnel
