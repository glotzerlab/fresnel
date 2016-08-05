// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel { namespace cpu {

/*! \param device Device to attach the Scene to
 *  Creates a new RTCScene and attaches it to the given device.
*/
Scene::Scene(std::shared_ptr<Device> device) : m_device(device)
    {
    RTCScene scene = rtcDeviceNewScene(device->getRTCDevice(), RTC_SCENE_DYNAMIC, RTC_INTERSECT1);

    if (scene == nullptr)
        {
        throw std::runtime_error("Error creating embree scene");
        }
    }

/*! Destroys the underlying RTCScene
 */
Scene::~Scene()
    {
    rtcDeleteScene(m_scene);
    }

/*! \param m Python module to export in
 */
void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene>(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device> >())
        ;
    }

} } // end namespace fresnel::cpu
