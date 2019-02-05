// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryPrism.h"

namespace fresnel { namespace gpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the polygon (in counterclockwise order)
    \param position position of each prism
    \param orientation orientation angle of each prism
    \param height height of each prism

    Initialize the prism.
*/
GeometryPrism::GeometryPrism(std::shared_ptr<Scene> scene,
                             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                             unsigned int N)
    : Geometry(scene)
    {
    // create the geometry
    // intersection and bounding programs are not stored for later destruction, as Device will destroy its program cache
    optix::Program intersection_program;
    optix::Program bounding_box_program;

    auto device = scene->getDevice();
    auto context = device->getContext();
    m_geometry = context->createGeometry();
    m_geometry->setPrimitiveCount(N);

    // load bounding and intresection programs
    const char * path_to_ptx = "_ptx_generated_GeometryPrism.cu.ptx";
    bounding_box_program = device->getProgram(path_to_ptx, "bounds");
    m_geometry->setBoundingBoxProgram(bounding_box_program);

    intersection_program = device->getProgram(path_to_ptx, "intersect");
    m_geometry->setIntersectionProgram(intersection_program);

    // set up the planes. The first two planes are the top (z=height) and bottom (z=0).
    // initialize both to 0 here, and other code will set the height appropriately
    std::vector< vec3<float> > plane_origin;
    std::vector< vec3<float> > plane_normal;
    float radius=0;

    plane_origin.push_back(vec3<float>(0,0,0));
    plane_normal.push_back(vec3<float>(0,0,1));
    plane_origin.push_back(vec3<float>(0,0,0));
    plane_normal.push_back(vec3<float>(0,0,-1));

    // now create planes for each of the polygon edges
    pybind11::buffer_info info = vertices.request();

    if (info.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info.shape[1] != 2)
        throw std::runtime_error("vertices must be a Nvert by 2 array");

    float *verts_f = (float *)info.ptr;

    for (unsigned int i = 0; i < info.shape[0]; i++)
        {
        // construct the normal and origin of each plane
        vec2<float> p0(verts_f[i*2], verts_f[i*2+1]);
        int j = (i + 1) % info.shape[0];
        vec2<float> p1(verts_f[j*2], verts_f[j*2+1]);
        vec2<float> n = -perp(p1 - p0);
        n = n / sqrtf(dot(n,n));

        plane_origin.push_back(vec3<float>(p0.x, p0.y, 0));
        plane_normal.push_back(vec3<float>(n.x, n.y, 0));

        // validate winding order
        int k = (j + 1) % info.shape[0];
        vec2<float> p2(verts_f[k*2], verts_f[k*2+1]);

        if (perpdot(p1 - p0, p2 - p1) <= 0)
            throw std::invalid_argument("vertices must be counterclockwise and convex");

        // precompute radius in the xy plane
        radius = std::max(radius, sqrtf(dot(p0,p0)));
        }

    // save radius for later use on the host side
    m_radius = radius;

    // copy data values to OptiX
    m_geometry["prism_radius"]->setFloat(radius);

    // set up OptiX data buffers
    m_plane_origin = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, plane_origin.size());
    m_plane_normal = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, plane_normal.size());

    vec3<float>* optix_plane_origin = (vec3<float>*)m_plane_origin->map();
    vec3<float>* optix_plane_normal = (vec3<float>*)m_plane_normal->map();

    for (unsigned int i = 0; i < plane_origin.size(); i++)
        {
        optix_plane_origin[i] = plane_origin[i];
        optix_plane_normal[i] = plane_normal[i];
        }

    m_plane_origin->unmap();
    m_plane_normal->unmap();

    m_geometry["prism_plane_origin"]->setBuffer(m_plane_origin);
    m_geometry["prism_plane_normal"]->setBuffer(m_plane_normal);

    optix::Buffer optix_position = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT2, N);
    optix::Buffer optix_angle = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
    optix::Buffer optix_height = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, N);
    optix::Buffer optix_color = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, N);

    m_geometry["prism_position"]->setBuffer(optix_position);
    m_geometry["prism_angle"]->setBuffer(optix_angle);
    m_geometry["prism_height"]->setBuffer(optix_height);
    m_geometry["prism_color"]->setBuffer(optix_color);

    // intialize python access to buffers
    m_position = std::make_shared< Array< vec2<float> > >(1, optix_position);
    m_angle = std::make_shared< Array< float > >(1, optix_angle);
    m_height = std::make_shared< Array< float > >(1, optix_height);
    m_color = std::make_shared< Array< RGB<float> > >(1, optix_color);
    setupInstance();

    m_valid = true;
    }

GeometryPrism::~GeometryPrism()
    {
    m_plane_origin->destroy();
    m_plane_normal->destroy();
    }

/*! \param m Python module to export in
 */
void export_GeometryPrism(pybind11::module& m)
    {
    pybind11::class_<GeometryPrism, std::shared_ptr<GeometryPrism> >(m, "GeometryPrism", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryPrism::getPositionBuffer)
        .def("getHeightBuffer", &GeometryPrism::getHeightBuffer)
        .def("getAngleBuffer", &GeometryPrism::getAngleBuffer)
        .def("getColorBuffer", &GeometryPrism::getColorBuffer)
        .def("getRadius", &GeometryPrism::getRadius)
        ;
    }

} } // end namespace fresnel::cpu
