// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometryCylinder.h"
#include "common/IntersectCylinder.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of cylinders to manage

    Initialize the cylinder geometry.
*/
GeometryCylinder::GeometryCylinder(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
    {
    // create the geometry
    m_geometry = rtcNewGeometry(m_device->getRTCDevice(), RTC_GEOMETRY_TYPE_USER);
    m_device->checkError();
    rtcSetGeometryUserPrimitiveCount(m_geometry, N);
    m_device->checkError();
    m_geom_id = rtcAttachGeometry(m_scene->getRTCScene(), m_geometry);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));

    // initialize the buffers
    m_points = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(2, N));
    m_radius = std::shared_ptr<Array<float>>(new Array<float>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(2, N));

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryCylinder::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryCylinder::intersect);
    m_device->checkError();

    m_valid = true;
    }

GeometryCylinder::~GeometryCylinder() { }

/*! Compute the bounding box of a given primitive

    \param args Arguments to the bounds check
*/
void GeometryCylinder::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometryCylinder* geom = (GeometryCylinder*)args->geometryUserPtr;
    const vec3<float> A = geom->m_points->get(args->primID * 2 + 0);
    const vec3<float> B = geom->m_points->get(args->primID * 2 + 1);
    const float radius = geom->m_radius->get(args->primID);

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = std::min(A.x - radius, B.x - radius);
    bounds_o.lower_y = std::min(A.y - radius, B.y - radius);
    bounds_o.lower_z = std::min(A.z - radius, B.z - radius);

    bounds_o.upper_x = std::max(A.x + radius, B.x + radius);
    bounds_o.upper_y = std::max(A.y + radius, B.y + radius);
    bounds_o.upper_z = std::max(A.z + radius, B.z + radius);
    }

/*! Compute the intersection of a ray with the given primitive

    \param args Arguments to the bounds check
*/
void GeometryCylinder::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    GeometryCylinder* geom = (GeometryCylinder*)args->geometryUserPtr;
    const vec3<float> A = geom->m_points->get(args->primID * 2 + 0);
    const vec3<float> B = geom->m_points->get(args->primID * 2 + 1);
    const float radius = geom->m_radius->get(args->primID);

    RTCRayHit& rayhit = *(RTCRayHit*)args->rayhit;
    RTCRay& ray = rayhit.ray;

    float t = HUGE_VALF, d = HUGE_VALF;
    vec3<float> N;
    unsigned int color_index;
    bool hit = intersect_ray_spherocylinder(t,
                                            d,
                                            N,
                                            color_index,
                                            vec3<float>(ray.org_x, ray.org_y, ray.org_z),
                                            vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z),
                                            A,
                                            B,
                                            radius);

    if (hit && (ray.tnear < t) && (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = (unsigned int)args->primID;
        rayhit.hit.Ng_x = N.x;
        rayhit.hit.Ng_y = N.y;
        rayhit.hit.Ng_z = N.z;

        FresnelRTCIntersectContext& context = *(FresnelRTCIntersectContext*)args->context;
        rayhit.hit.instID[0] = context.context.instID[0];
        context.shading_color = geom->m_color->get(args->primID * 2 + color_index);
        context.d = d;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryCylinder(pybind11::module& m)
    {
    pybind11::class_<GeometryCylinder, Geometry, std::shared_ptr<GeometryCylinder>>(
        m,
        "GeometryCylinder")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPointsBuffer", &GeometryCylinder::getPointsBuffer)
        .def("getRadiusBuffer", &GeometryCylinder::getRadiusBuffer)
        .def("getColorBuffer", &GeometryCylinder::getColorBuffer);
    }

    } // namespace cpu
    } // namespace fresnel
