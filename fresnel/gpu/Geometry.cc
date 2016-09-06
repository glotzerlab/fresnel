// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Geometry.h"

namespace fresnel { namespace gpu {

/*! \param scene Scene to attach the Geometry to
    The base class constructor does nothing. It is up to the derived classes to implement appropriate
    geometry creation routines and set m_valid and m_geom_id appropriately.
*/
Geometry::Geometry(std::shared_ptr<Scene> scene) : m_scene(scene), m_device(scene->getDevice())
    {
    std::cout << "Create GPU Geometry" << std::endl;
    }

Geometry::~Geometry()
    {
    std::cout << "Destroy GPU Geometry" << std::endl;
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
    // TODO: set up material

    m_valid = true;
    enable();
    }
/*! \param m Python module to export in
 */
void export_Geometry(pybind11::module& m)
    {
    pybind11::class_<Geometry, std::shared_ptr<Geometry> >(m, "Geometry")
        .def(pybind11::init<std::shared_ptr<Scene> >())
        .def("getMaterial", &Geometry::getMaterial)
        .def("setMaterial", &Geometry::setMaterial)
        .def("disable", &Geometry::disable)
        .def("enable", &Geometry::enable)
        .def("remove", &Geometry::remove)
        ;
    }

} } // end namespace fresnel::gpu
