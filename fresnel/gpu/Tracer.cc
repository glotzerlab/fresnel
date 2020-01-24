// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Tracer.h"

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
 */
Tracer::Tracer(std::shared_ptr<Device> device, unsigned int w, unsigned int h) : m_device(device)
{
    resize(w, h);
    m_highlight_warning = false;
    m_highlight_warning_color = RGB<float>(1, 0, 1);
}

Tracer::~Tracer() {}

/*! \param w New output buffer width
    \param h New output buffer height

    Delete the old output buffer and make a new one with the given width and height
*/
void Tracer::resize(unsigned int w, unsigned int h)
{
    if (w == 0 || h == 0)
        throw std::runtime_error("Invalid dimensions");

    m_linear_out_gpu
        = m_device->getContext()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, w, h);
    m_srgb_out_gpu
        = m_device->getContext()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, w, h);

    m_linear_out_gpu->setSize(w, h);
    m_srgb_out_gpu->setSize(w, h);
    m_w = w;
    m_h = h;

    void* tmp = m_linear_out_gpu->map();
    memset(tmp, 0, m_w * m_h * 16);
    m_linear_out_gpu->unmap();

    tmp = m_srgb_out_gpu->map();
    memset(tmp, 0, m_w * m_h * 4);
    m_srgb_out_gpu->unmap();

    m_linear_out_py = std::make_shared<Array<RGBA<float>>>(2, m_linear_out_gpu);
    m_srgb_out_py = std::make_shared<Array<RGBA<unsigned char>>>(2, m_srgb_out_gpu);
}

/*! \param scene The Scene to render
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
    pybind11::class_<Tracer, std::shared_ptr<Tracer>>(m, "Tracer")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int>())
        .def("render", &Tracer::render)
        .def("resize", &Tracer::resize)
        .def("getSRGBOutputBuffer", &Tracer::getSRGBOutputBuffer)
        .def("getLinearOutputBuffer", &Tracer::getLinearOutputBuffer)
        .def("enableHighlightWarning", &Tracer::enableHighlightWarning)
        .def("disableHighlightWarning", &Tracer::disableHighlightWarning)
        .def("getSeed", &Tracer::getSeed)
        .def("setSeed", &Tracer::setSeed);
}

}} // end namespace fresnel::gpu
