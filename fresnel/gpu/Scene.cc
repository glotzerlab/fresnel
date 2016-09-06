// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel { namespace gpu {

/*! \param device Device to attach the Scene to
 *  Creates a new geometry group.
*/
Scene::Scene(std::shared_ptr<Device> device) : m_device(device)
    {
    std::cout << "Create GPU Scene" << std::endl;
    m_root = m_device->getContext()->createGeometryGroup();
    m_accel = m_device->getContext()->createAcceleration("Trbvh");
    m_root->setAcceleration(m_accel);
    }

/*! Destroys the root scene object.
 */
Scene::~Scene()
    {
    std::cout << "Destroy GPU Scene" << std::endl;
    m_accel->destroy();
    m_root->destroy();
    }

/*! \param m Python module to export in
 */
void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene, std::shared_ptr<Scene> >(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device> >())
        ;
    }

} } // end namespace fresnel::gpu
