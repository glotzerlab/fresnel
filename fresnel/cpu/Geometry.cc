// Copyright (c) 2016-2021 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Geometry.h"
#include <embree3/rtcore.h>

namespace fresnel
    {
namespace cpu
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
        rtcEnableGeometry(m_geometry);
        m_device->checkError();
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
        rtcDisableGeometry(m_geometry);
        m_device->checkError();
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
        rtcDetachGeometry(m_scene->getRTCScene(), m_geom_id);
        rtcReleaseGeometry(m_geometry);
        m_valid = false;
        m_device->checkError();
        }
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

    } // namespace cpu
    } // namespace fresnel
