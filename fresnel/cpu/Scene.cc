// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel { namespace cpu {

Scene::Scene(std::shared_ptr<Device> device) : m_device(device)
    {
    RTCScene scene = rtcDeviceNewScene(device->getDevice(), RTC_SCENE_DYNAMIC, RTC_INTERSECT1);

    if (scene == nullptr)
        {
        throw std::runtime_error("Error creating embree scene");
        }
    }

Scene::~Scene()
    {
    rtcDeleteScene(m_scene);
    }

void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene>(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device> >())
        ;
    }

} } // end namespace fresnel::cpu
