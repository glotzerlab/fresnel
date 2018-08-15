// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.
#include <stdexcept>
#include <pybind11/stl.h>

#include "GeometryMesh.h"
#include "common/IntersectTriangle.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param N number of mesh triangles to manage

    Initialize the mesh geometry.
*/
GeometryMesh::GeometryMesh(std::shared_ptr<Scene> scene, unsigned int N)
    : Geometry(scene)
    {
    // create the geometry
    m_geometry = rtcNewGeometry(m_device->getRTCDevice(), RTC_GEOMETRY_TYPE_USER);
    m_device->checkError();
    rtcSetGeometryUserPrimitiveCount(m_geometry,N);
    m_device->checkError();
    m_geom_id = rtcAttachGeometry(m_scene->getRTCScene(), m_geometry);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1,0,1)));
    setOutlineMaterial(Material(RGB<float>(0,0,0), 1.0f));

    // initialize the buffers
    m_points = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(3,N));
    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(3,N));

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryMesh::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryMesh::intersect);
    m_device->checkError();

    m_valid = true;
    }

GeometryMesh::~GeometryMesh()
    {
    }

/*! Compute the bounding box of a given primitive

    \param args Arguments to the bounds check
*/
void GeometryMesh::bounds(const struct RTCBoundsFunctionArguments *args)
    {
    GeometryMesh *geom = (GeometryMesh*)args->geometryUserPtr;
    const vec3<float> a = geom->m_points->get(args->primID*3 + 0);
    const vec3<float> b = geom->m_points->get(args->primID*3 + 1);
    const vec3<float> c = geom->m_points->get(args->primID*3 + 2);

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = std::min(std::min(a.x, b.x), c.x);
    bounds_o.lower_y = std::min(std::min(a.y, b.y), c.y);
    bounds_o.lower_z = std::min(std::min(a.z, b.z), c.z);

    bounds_o.upper_x = std::max(std::max(a.x, b.x), c.x);
    bounds_o.upper_y = std::max(std::max(a.y, b.y), c.y);
    bounds_o.upper_z = std::max(std::max(a.z, b.z), c.z);

    //std::cerr << "Debug printing" << std::endl;
    //std::cerr << bounds_o.upper_x << std::endl;

    }

/*! Compute the intersection of a ray with the given primitive

    \param args Arguments to the intersect check
*/
void GeometryMesh::intersect(const struct RTCIntersectFunctionNArguments *args)
    {
    GeometryMesh *geom = (GeometryMesh*)args->geometryUserPtr;
    const vec3<float> a = geom->m_points->get(args->primID*3 + 0);
    const vec3<float> b = geom->m_points->get(args->primID*3 + 1);
    const vec3<float> c = geom->m_points->get(args->primID*3 + 2);

    RTCRayHit& rayhit = *(RTCRayHit *)args->rayhit;
    RTCRay& ray = rayhit.ray;

    float t=HUGE_VALF, d=HUGE_VALF;
    vec3<float> N;
    vec3<float> color_index;
    bool hit = intersect_ray_triangle(t,
                                      d,
                                      N,
                                      color_index,
                                      vec3<float>(ray.org_x,ray.org_y,ray.org_z),
                                      vec3<float>(ray.dir_x,ray.dir_y,ray.dir_z),
                                      a,
                                      b,
                                      c);
    //std::cerr << "Debug printing" << std::endl;
    //std::cerr << hit << std::endl;

    if (hit && (ray.tnear < t) && (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = (unsigned int) args->primID;
        rayhit.hit.Ng_x = N.x;
        rayhit.hit.Ng_y = N.y;
        rayhit.hit.Ng_z = N.z;

        FresnelRTCIntersectContext & context = *(FresnelRTCIntersectContext *)args->context;
        rayhit.hit.instID[0] = context.context.instID[0];
        context.shading_color = geom->m_color->get(args->primID*3 + 0)*color_index.x + geom->m_color->get(args->primID*3 + 1)*color_index.y + geom->m_color->get(args->primID*3 + 2)*color_index.z;
        context.d = d;
        }

    }

/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, std::shared_ptr<GeometryMesh> >(m, "GeometryMesh", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPointsBuffer", &GeometryMesh::getPointsBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer)
        ;
    }

} } // end namespace fresnel::cpu
