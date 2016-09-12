// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include "Tracer.h"

namespace fresnel { namespace gpu {

/*! \param device Device to attach the raytracer to
*/
Tracer::Tracer(std::shared_ptr<Device> device, unsigned int w, unsigned int h)
    : m_device(device)
    {
    std::cout << "Create GPU Tracer" << std::endl;
    resize(w,h);
    }

Tracer::~Tracer()
    {
    std::cout << "Destroy GPU Tracer" << std::endl;
    m_ray_gen->destroy();
    }

/*! \param w New output buffer width
    \param h New output buffer height

    Delete the old output buffer and make a new one with the given width and height
*/
void Tracer::resize(unsigned int w, unsigned int h)
    {
    if (w == 0 || h == 0)
        throw std::runtime_error("Invalid dimensions");

    m_out = std::unique_ptr< RGBA<float>[] >(new RGBA<float>[w*h]);
    m_w = w;
    m_h = h;
    }

/*! \param scene The Scene to render
*/
void Tracer::render(std::shared_ptr<Scene> scene)
    {
    if (scene->getDevice() != m_device)
        throw std::runtime_error("Scene and Tracer devices do not match");
    }

/*! \param camera The camera to set.
*/
void Tracer::setCamera(const Camera& camera)
    {
    m_camera = camera;
    }

pybind11::buffer_info Tracer::getBuffer()
    {
    // TODO: Need to implement buffer objects, gamma correction, and copy data from the GPU to the CPU
    std::cout << "Tracer::getBuffer" << std::endl;
    return pybind11::buffer_info(m_out.get(),                               /* Pointer to buffer */
                                 sizeof(float),                             /* Size of one scalar */
                                 pybind11::format_descriptor<float>::value, /* Python struct-style format descriptor */
                                 3,                                         /* Number of dimensions */
                                 { m_h, m_w, 4 },                           /* Buffer dimensions */
                                 { sizeof(float)*m_w*4,                     /* Strides (in bytes) for each index */
                                   sizeof(float)*4,
                                   sizeof(float)
                                 });
    }

/*! \param m Python module to export in
 */
void export_Tracer(pybind11::module& m)
    {
    pybind11::class_<Tracer, std::shared_ptr<Tracer> >(m, "Tracer")
        .def(pybind11::init<std::shared_ptr<Device>, unsigned int, unsigned int >())
        .def("render", &Tracer::render)
        .def("resize", &Tracer::resize)
        .def("setCamera", &Tracer::setCamera)
        .def_buffer([](Tracer &t) -> pybind11::buffer_info { return t.getBuffer(); })
        ;
    }

} } // end namespace fresnel::gpu
