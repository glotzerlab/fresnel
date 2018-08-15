// Copyright (c) 2016-2018 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <string>
#include <iostream>

#include "TracerPath.h"
#include "TracerIDs.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
*/
TracerPath::TracerPath(std::shared_ptr<Device> device, unsigned int w, unsigned int h, unsigned int light_samples)
    : Tracer(device, w, h), m_light_samples(light_samples)
    {
    // create the entry point program
    optix::Context context = m_device->getContext();
    m_ray_gen = m_device->getProgram("_ptx_generated_path.cu.ptx", "path_ray_gen");
    m_ray_gen_entry = m_device->getEntryPoint("_ptx_generated_path.cu.ptx", "path_ray_gen");

    // load the exception program
    m_exception_program = m_device->getProgram("_ptx_generated_path.cu.ptx", "path_exception");
    context->setExceptionProgram(m_ray_gen_entry, m_exception_program);
    reset();
    }

TracerPath::~TracerPath()
    {
    }

void TracerPath::reset()
    {
    m_n_samples = 0;
    m_seed++;

    void *tmp = m_linear_out_gpu->map();
    memset(tmp, 0, m_w*m_h * 16);
    m_linear_out_gpu->unmap();

    tmp = m_srgb_out_gpu->map();
    memset(tmp, 0, m_w*m_h * 4);
    m_srgb_out_gpu->unmap();
    }

//! Initialize the Material for use in tracing
void TracerPath::setupMaterial(optix::Material mat, Device *dev)
    {
    optix::Program p = dev->getProgram("_ptx_generated_path.cu.ptx", "path_closest_hit");
    mat->setClosestHitProgram(TRACER_PATH_RAY_ID, p);
    }

/*! \param scene The Scene to render
*/
void TracerPath::render(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    const Camera camera(scene->getCamera());
    const Lights lights(scene->getLights(), camera);

    Tracer::render(scene);

    // update number of samples (the first sample is 1)
    m_n_samples++;

    optix::Context context = m_device->getContext();

    // set common variables before launch
    context["top_object"]->set(scene->getRoot());
    context["scene_epsilon"]->setFloat(1.e-3f);
    context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
    context["srgb_output_buffer"]->set(m_srgb_out_gpu);
    context["linear_output_buffer"]->set(m_linear_out_gpu);

    // set camera variables
    const Camera cam(scene->getCamera());
    context["cam"]->setUserData(sizeof(cam), &cam);
    context["lights"]->setUserData(sizeof(lights), &lights);

    // set background variables
    context["background_color"]->setUserData(sizeof(background_color), &background_color);
    context["background_alpha"]->setFloat(background_alpha);

    // set highlight warning
    context["highlight_warning_color"]->setUserData(sizeof(m_highlight_warning_color), &m_highlight_warning_color);
    context["highlight_warning"]->setUint(m_highlight_warning);

    // path tracer settings
    context["seed"]->setUint(m_seed);
    context["n_samples"]->setUint(m_n_samples);
    context["light_samples"]->setUint(m_light_samples);

    // TODO: Consider using progressive launches to better utilize multi-gpu systems
    context->launch(m_ray_gen_entry, m_w, m_h);
    }

/*! \param m Python module to export in
 */
void export_TracerPath(pybind11::module& m)
    {
    pybind11::class_<TracerPath, std::shared_ptr<TracerPath> >(m, "TracerPath", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int, unsigned int>())
        .def("getNumSamples", &TracerPath::getNumSamples)
        .def("reset", &TracerPath::reset)
        .def("setLightSamples", &TracerPath::setLightSamples)
        ;
    }

} } // end namespace fresnel::gpu
