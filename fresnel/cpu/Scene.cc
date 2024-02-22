// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "Scene.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param device Device to attach the Scene to
 *  Creates a new RTCScene and attaches it to the given device.
 */
Scene::Scene(std::shared_ptr<Device> device)
    : m_device(device), m_background_color(RGB<float>(0, 0, 0)), m_background_alpha(0.0)
    {
    m_scene = rtcNewScene(device->getRTCDevice());
    rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_LOW);
    m_device->checkError();

    m_lights.N = 2;
    m_lights.direction[0] = vec3<float>(-1, 0.3, 1);
    m_lights.color[0] = RGB<float>(1, 1, 1);

    m_lights.direction[1] = vec3<float>(1, 0, 1);
    m_lights.color[1] = RGB<float>(0.1, 0.1, 0.1);
    }

/*! Destroys the underlying RTCScene
 */
Scene::~Scene()
    {
    rtcReleaseScene(m_scene);
    m_device->checkError();
    }

/*! \param m Python module to export in
 */
void export_Scene(pybind11::module& m)
    {
    pybind11::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
        .def(pybind11::init<std::shared_ptr<Device>>())
        .def("setCamera", &Scene::setCamera)
        .def("getCamera", &Scene::getCamera, pybind11::return_value_policy::reference_internal)
        .def("getBackgroundColor", &Scene::getBackgroundColor)
        .def("setBackgroundColor", &Scene::setBackgroundColor)
        .def("getBackgroundAlpha", &Scene::getBackgroundAlpha)
        .def("setBackgroundAlpha", &Scene::setBackgroundAlpha)
        .def("getLights", &Scene::getLights, pybind11::return_value_policy::reference_internal)
        .def("setLights", &Scene::setLights);
    }

    } // namespace cpu
    } // namespace fresnel
