// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <string>

#include "TracerDirect.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
*/
TracerDirect::TracerDirect(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    // create the entry point program
    optix::Context context = m_device->getContext();
    m_ray_gen = m_device->getProgram("_ptx_generated_direct.cu.ptx", "direct_ray_gen");
    m_ray_gen_entry = m_device->getEntryPoint("_ptx_generated_direct.cu.ptx", "direct_ray_gen");

    // load the exception program
    m_exception_program = m_device->getProgram("_ptx_generated_direct.cu.ptx", "direct_exception");
    context->setExceptionProgram(m_ray_gen_entry, m_exception_program);
    }

TracerDirect::~TracerDirect()
    {
    }

//! Initialize the Material for use in tracing
void TracerDirect::setupMaterial(optix::Material mat, Device *dev)
    {
    optix::Program p = dev->getProgram("_ptx_generated_direct.cu.ptx", "direct_closest_hit");
    mat->setClosestHitProgram(0, p);
    }

/*! \param scene The Scene to render
*/
void TracerDirect::render(std::shared_ptr<Scene> scene)
    {
    const RGB<float> background_color = scene->getBackgroundColor();
    const float background_alpha = scene->getBackgroundAlpha();
    const vec3<float> light_direction = scene->getLightDirection();

    const Camera camera(scene->getCamera());

    Tracer::render(scene);

    optix::Context context = m_device->getContext();

    // set common variables before launch
    context["top_object"]->set(scene->getRoot());
    context["scene_epsilon"]->setFloat(1.e-3f);
    context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
    context["srgb_output_buffer"]->set(m_srgb_out_gpu);
    context["linear_output_buffer"]->set(m_linear_out_gpu);

    // set camera variables
    context["camera_p"]->setFloat(camera.p.x, camera.p.y, camera.p.z);
    context["camera_d"]->setFloat(camera.d.x, camera.d.y, camera.d.z);
    context["camera_u"]->setFloat(camera.u.x, camera.u.y, camera.u.z);
    context["camera_r"]->setFloat(camera.r.x, camera.r.y, camera.r.z);
    context["camera_h"]->setFloat(camera.h);

    // set background variables
    context["background_color"]->setFloat(background_color.r, background_color.g, background_color.b);
    context["background_alpha"]->setFloat(background_alpha);
    context["light_direction"]->setFloat(light_direction.x, light_direction.y, light_direction.z);

    context->launch(m_ray_gen_entry, m_w, m_h);
    }

/*! \param m Python module to export in
 */
void export_TracerDirect(pybind11::module& m)
    {
    pybind11::class_<TracerDirect, std::shared_ptr<TracerDirect> >(m, "TracerDirect", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        ;
    }

} } // end namespace fresnel::gpu
