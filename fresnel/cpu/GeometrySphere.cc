// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometrySphere.h"
#include "common/IntersectSphere.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of spheres to manage

    Initialize the sphere geometry.
*/
GeometrySphere::GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
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
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_radius = std::shared_ptr<Array<float>>(new Array<float>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometrySphere::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometrySphere::intersect);
    m_device->checkError();

    rtcCommitGeometry(m_geometry);
    m_device->checkError();

    m_valid = true;
    }

GeometrySphere::~GeometrySphere() { }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometrySphere instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometrySphere::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometrySphere* geom = (GeometrySphere*)args->geometryUserPtr;
    vec3<float> p = geom->m_position->get(args->primID);
    float radius = geom->m_radius->get(args->primID);
    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = p.x - radius;
    bounds_o.lower_y = p.y - radius;
    bounds_o.lower_z = p.z - radius;

    bounds_o.upper_x = p.x + radius;
    bounds_o.upper_y = p.y + radius;
    bounds_o.upper_z = p.z + radius;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometrySphere instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometrySphere::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    *args->valid = 0;

    GeometrySphere* geom = (GeometrySphere*)args->geometryUserPtr;
    const vec3<float> position = geom->m_position->get(args->primID);
    const float radius = geom->m_radius->get(args->primID);
    RTCRayHit& rayhit = *(RTCRayHit*)args->rayhit;
    RTCRay& ray = rayhit.ray;

    float t = 0, d = 0;
    vec3<float> N;
    bool hit = intersect_ray_sphere(t,
                                    d,
                                    N,
                                    vec3<float>(ray.org_x, ray.org_y, ray.org_z),
                                    vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z),
                                    position,
                                    radius);

    if (hit && (ray.tnear < t) && (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = (unsigned int)args->primID;
        rayhit.hit.Ng_x = ray.org_x + t * ray.dir_x - position.x;
        rayhit.hit.Ng_y = ray.org_y + t * ray.dir_y - position.y;
        rayhit.hit.Ng_z = ray.org_z + t * ray.dir_z - position.z;
        FresnelRTCIntersectContext& context = *(FresnelRTCIntersectContext*)args->context;
        rayhit.hit.instID[0] = context.context.instID[0];
        context.shading_color = geom->m_color->get(args->primID);
        context.d = d;
        *args->valid = -1;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometrySphere(pybind11::module& m)
    {
    pybind11::class_<GeometrySphere, Geometry, std::shared_ptr<GeometrySphere>>(m, "GeometrySphere")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometrySphere::getPositionBuffer)
        .def("getRadiusBuffer", &GeometrySphere::getRadiusBuffer)
        .def("getColorBuffer", &GeometrySphere::getColorBuffer);
    }

    } // namespace cpu
    } // namespace fresnel
