// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Whitted.h"

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
Whitted::Whitted(std::shared_ptr<Device> device) : m_device(device)
    {
    }

Whitted::~Whitted()
    {
    }

void Whitted::render(std::shared_ptr<Scene> scene, unsigned int w, unsigned int h)
    {
    rtcCommit(scene->getRTCScene());
    m_device->checkError();
    }


/*! \param m Python module to export in
 */
void export_Whitted(pybind11::module& m)
    {
    pybind11::class_<Whitted, std::shared_ptr<Whitted> >(m, "Whitted")
        .def(pybind11::init<std::shared_ptr<Device> >())
        .def("render", &Whitted::render)
        ;
    }

} } // end namespace fresnel::cpu
