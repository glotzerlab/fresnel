// Copyright (c) 2016 The Regents of the University of Michigan
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
    const string& ptx_root = m_device->getPTXRoot();
    m_ray_gen = context->createProgramFromPTXFile(ptx_root + "/whitted.ptx", "whitted_ray_gen");

    m_ray_gen_entry = context->getEntryPointCount();
    context->setEntryPointCount(m_ray_gen_entry + 1);
    context->setRayGenerationProgram(m_ray_gen_entry, m_ray_gen);
    }

TracerWhitted::~TracerWhitted()
    {
    std::cout << "Destroy GPU TracerWhitted" << std::endl;
    }

/*! \param m Python module to export in
 */
void export_TracerWhitted(pybind11::module& m)
    {
    pybind11::class_<TracerWhitted, std::shared_ptr<TracerWhitted> >(m, "TracerWhitted", pybind11::base<Tracer>())
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def_buffer([](TracerWhitted &t) -> pybind11::buffer_info { return t.getBuffer(); })  // repeated because def_buffer does not inherit
        ;
    }

} } // end namespace fresnel::gpu
