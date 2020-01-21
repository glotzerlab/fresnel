// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <string>

#include "TracerDirect.h"
#include "TracerIDs.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
*/
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    // create the entry point program
    optix::Context context = m_device->getContext();
    m_ray_gen = m_device->getProgram("direct.ptx", "direct_ray_gen");
    m_ray_gen_entry = m_device->getEntryPoint("direct.ptx", "direct_ray_gen");

    // load the exception program
    m_exception_program = m_device->getProgram("direct.ptx", "direct_exception");
    context->setExceptionProgram(m_ray_gen_entry, m_exception_program);
    }

TracerDirect::~TracerDirect()
    {
    }

//! Initialize the Material for use in tracing
void TracerDirect::setupMaterial(optix::Material mat, Device *dev)
    {
    optix::Program p = dev->getProgram("direct.ptx", "direct_closest_hit");
    mat->setClosestHitProgram(TRACER_PREVIEW_RAY_ID, p);
    }

/*! \param scene The Scene to render
*/
void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();

    const Camera camera(scene->getCamera());
    const Lights lights(scene->getLights(), camera);

    Tracer::render(scene);

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

    // anti-aliasing settings
    context["aa_n"]->setUint(m_aa_n);
    context["seed"]->setUint(m_seed);

    context->launch(m_ray_gen_entry, m_w, m_h);
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, Tracer, std::shared_ptr<TracerDirect> >(m, "TracerDirect")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("setAntialiasingN", &TracerDirect::setAntialiasingN)
        .def("getAntialiasingN", &TracerDirect::getAntialiasingN)
        ;
    }

} } // end namespace fresnel::gpu
