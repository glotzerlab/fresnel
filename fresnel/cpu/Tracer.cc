// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Tracer.h"

namespace fresnel { namespace cpu {

/*! \param device Device to attach the raytracer to
*/
Tracer::Tracer(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : m_device(device)
    {
    std::cout << "Create Tracer" << std::endl;
    resize(w,h);
    }

Tracer::~Tracer()
    {
    std::cout << "Destroy Tracer" << std::endl;
    }

/*! \param w New output buffer width
    \param h New output buffer height

    Make a new output buffer with the given width and height
*/
void Tracer::resize(unsigned int w, unsigned int h)
    {
    if (w == 0 || h == 0)
        throw std::runtime_error("Invalid dimensions");

    m_out = std::shared_ptr< Array< RGBA<float> > >(new Array< RGBA<float> >(w, h));
    }

/*! \param scene The Scene to render

    Derived classes must implement this method.
*/
void Tracer::render(std::shared_ptr<Scene> scene)
    {
    if (scene->getDevice() != m_device)
        throw std::runtime_error("Scene and Tracer devices do not match");
    }

/*! \param m Python module to export in
 */
void export_Tracer(pybind11::module& m)
    {
    pybind11::class_<Tracer, std::shared_ptr<Tracer> >(m, "Tracer")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int >())
        .def("render", &Tracer::render)
        .def("resize", &Tracer::resize)
        .def("getOutputBuffer", &Tracer::getOutputBuffer)
        ;
    }

} } // end namespace fresnel::cpu
