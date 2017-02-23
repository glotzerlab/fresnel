// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel { namespace gpu {

/*! \param device Device to attach the Scene to
 *  Creates a new geometry group.
*/
Scene::Scene(std::shared_ptr<Device> device) : m_device(device), m_background_color(RGB<float>(0,0,0)), m_background_alpha(0.0)
    {
    m_root = m_device->getContext()->createGeometryGroup();
    m_accel = m_device->getContext()->createAcceleration("Trbvh");
    m_root->setAcceleration(m_accel);

    vec3<float> l = vec3<float>(0.2,1,0.5);
    l = l / sqrtf(dot(l,l));
    m_light_direction = l;
    }

/*! Destroys the root scene object.
 */
Scene::~Scene()
    {
    m_accel->destroy();
    m_root->destroy();
    }

/*! \param m Python module to export in
 */
void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene, std::shared_ptr<Scene> >(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device> >())
        .def("getCamera", &Scene::getCamera)
        .def("setCamera", &Scene::setCamera)
        .def("getBackgroundColor", &Scene::getBackgroundColor)
        .def("setBackgroundColor", &Scene::setBackgroundColor)
        .def("getBackgroundAlpha", &Scene::getBackgroundAlpha)
        .def("setBackgroundAlpha", &Scene::setBackgroundAlpha)
        .def("getLightDirection", &Scene::getLightDirection)
        .def("setLightDirection", &Scene::setLightDirection)
        ;
    }

} } // end namespace fresnel::gpu
