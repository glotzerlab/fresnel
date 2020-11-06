// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <pybind11/stl.h>
#include <stdexcept>

#include "GeometryEllipsoid.h"
#include "common/IntersectEllipsoid.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param N number of ellipsoids to manage

    Initialize the ellipsoid geometry.
*/
GeometryEllipsoid::GeometryEllipsoid(std::shared_ptr<Scene> scene, unsigned int N) : Geometry(scene)
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
    m_radii = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
	m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>);
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryEllipsoid::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryEllipsoid::intersect);
    m_device->checkError();

    rtcCommitGeometry(m_geometry);
    m_device->checkError();

    m_valid = true;
    }

GeometryEllipsoid::~GeometryEllipsoid() { }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryEllipsoid instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryEllipsoid::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometryEllipsoid* geom = (GeometryEllipsoid*)args->geometryUserPtr;
    vec3<float> p = geom->m_position->get(args->primID);
	vec3<float> radii = geom->m_radii->get(args->primID);

	// Use a too-big box for now
    float max_radius = std::max_element(radii.begin(), radii.end());
    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = p.x - max_radius;
    bounds_o.lower_y = p.y - max_radius;
    bounds_o.lower_z = p.z - max_radius;

    bounds_o.upper_x = p.x + max_radius;
    bounds_o.upper_y = p.y + max_radius;
    bounds_o.upper_z = p.z + max_radius;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryEllipsoid instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryEllipsoid::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    GeometryEllipsoid* geom = (GeometryEllipsoid*)args->geometryUserPtr;
    const vec3<float> position = geom->m_position->get(args->primID);
	vec3<float> radii = geom->m_radii->get(args->primID);
	quat<float> orientation = geom->m_orientation(args->primID)
    RTCRayHit& rayhit = *(RTCRayHit*)args->rayhit;
    RTCRay& ray = rayhit.ray;

    float t = 0, d = 0;
    vec3<float> N;
    bool hit = intersect_ray_ellipsoid(t,
									   d,
									   N,
									   vec3<float>(ray.org_x, ray.org_y, ray.org_z),
									   vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z),
									   position,
									   radii,
									   orientation
									  );

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
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryEllipsoid(pybind11::module& m)
    {
    pybind11::class_<GeometryEllipsoid, Geometry, std::shared_ptr<GeometryEllipsoid>>(m, "GeometryEllipsoid")
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometryEllipsoid::getPositionBuffer)
        .def("getRadiiBuffer", &GeometryEllipsoid::getRadiiBuffer)
		.def("getOrientationBuffer", &GeometryEllipsoid::getOrientationBuffer)
        .def("getColorBuffer", &GeometryEllipsoid::getColorBuffer);
    }

    } // namespace cpu
    } // namespace fresnel
