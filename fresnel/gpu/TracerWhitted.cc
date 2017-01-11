// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <string>

#include "TracerWhitted.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
*/
TracerWhitted::TracerWhitted(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : Tracer(device, w, h)
    {
    std::cout << "Create GPU TracerWhitted" << std::endl;

    // create the entry point program
    optix::Context context = m_device->getContext();
    m_ray_gen = m_device->getProgram("_ptx_generated_whitted.cu.ptx", "whitted_ray_gen");
    m_ray_gen_entry = m_device->getEntryPoint("_ptx_generated_whitted.cu.ptx", "whitted_ray_gen");

    // load the exception program
    m_exception_program = m_device->getProgram("_ptx_generated_whitted.cu.ptx", "whitted_exception");
    context->setExceptionProgram(m_ray_gen_entry, m_exception_program);
    }

TracerWhitted::~TracerWhitted()
    {
    std::cout << "Destroy GPU TracerWhitted" << std::endl;
    }

//! Initialize the Material for use in tracing
void TracerWhitted::setupMaterial(optix::Material mat, Device *dev)
    {
    optix::Program p = dev->getProgram("_ptx_generated_whitted.cu.ptx", "whitted_closest_hit");
    mat->setClosestHitProgram(0, p);
    }

/*! \param scene The Scene to render
*/
void TracerWhitted::render(std::shared_ptr<Scene> scene)
    {
    Tracer::render(scene);

    optix::Context context = m_device->getContext();

    // set common variables before launch
    context["top_object"]->set(scene->getRoot());
    context["scene_epsilon"]->setFloat(1.e-3f);
    context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
    context["output_buffer"]->set(m_out_gpu);

    // set camera variables
    context["camera_p"]->setFloat(m_camera.p.x, m_camera.p.y, m_camera.p.z);
    context["camera_d"]->setFloat(m_camera.d.x, m_camera.d.y, m_camera.d.z);
    context["camera_u"]->setFloat(m_camera.u.x, m_camera.u.y, m_camera.u.z);
    context["camera_r"]->setFloat(m_camera.r.x, m_camera.r.y, m_camera.r.z);
    context["camera_h"]->setFloat(m_camera.h);

    context->launch(m_ray_gen_entry, m_w, m_h);
    }

/*! \param m Python module to export in
 */
void export_TracerWhitted(pybind11::module& m)
    {
    pybind11::class_<TracerWhitted, std::shared_ptr<TracerWhitted> >(m, "TracerWhitted", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        ;
    }

} } // end namespace fresnel::gpu
