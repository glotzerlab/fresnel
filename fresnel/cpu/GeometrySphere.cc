// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>

#include "GeometrySphere.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param N number of spheres to manage

    Initialize the sphere geometry.
*/
GeometrySphere::GeometrySphere(std::shared_ptr<Scene> scene, unsigned int N)
    : Geometry(scene)
    {
    // create the geometry
    RTCGeometry geometry = rtcNewGeometry(m_device->getRTCDevice(), RTC_GEOMETRY_TYPE_USER);
    m_geom_id = rtcAttachGeometry(m_scene->getRTCScene(), geometry);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1,0,1)));
    setOutlineMaterial(Material(RGB<float>(0,0,0), 1.0f));

    // initialize the buffers
    m_position = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_radius = std::shared_ptr< Array< float > >(new Array< float >(N));
    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));

    // register functions for embree
    rtcSetGeometryUserData(geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(geometry, &GeometrySphere::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(geometry, &GeometrySphere::intersect);
    m_device->checkError();
    rtcSetGeometryOccludedFunction(geometry, &GeometrySphere::occlude);
    m_device->checkError();

    m_valid = true;
    }

GeometrySphere::~GeometrySphere()
    {
    }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometrySphere instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometrySphere::bounds(const struct RTCBoundsFunctionArguments *args)
    {
    GeometrySphere *geom = (GeometrySphere*)args->geometryUserPtr;
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
void GeometrySphere::intersect(const struct RTCIntersectFunctionNArguments *args)
   {
    GeometrySphere *geom = (GeometrySphere*)args->geometryUserPtr;
    const vec3<float> position = geom->m_position->get(args->primID);
    RTCRayHit& rayhit = *(RTCRayHit *)args->rayhit;
    RTCRay& ray = rayhit.ray;
    const vec3<float> v = position-vec3<float>(ray.org_x, ray.org_y, ray.org_z);
    const float vsq = dot(v,v);
    const float radius = geom->m_radius->get(args->primID);
    const float rsq = (radius)*(radius);
    vec3<float> dir = vec3<float>(ray.dir_x,ray.dir_y, ray.dir_z);
    const vec3<float> w = cross(v,dir);
    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(dir,dir);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(vsq - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    float t;
    if (dot(v, dir) > 0.0f)
        {
        if (vsq > rsq)
            {
            // ray origin is outside the sphere, compute the distance back from the closest point
            t = Rp-Ri;
            }
        else
            {
            // ray origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Rp+Ri;
            }
        }
    else
        {
        // origin is behind the sphere (use tolerance to exclude origins directly on the sphere)
        if (vsq - rsq > -3e-6f*rsq)
            {
            // origin is outside the sphere, no intersection
            return;
            }
        else
            {
            // origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Ri-Rp;
            }
        }

    if ((ray.tnear < t) & (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = (unsigned int) args->primID;
        rayhit.hit.Ng_x = ray.org_x+t*ray.dir_x-position.x;
        rayhit.hit.Ng_y = ray.org_y+t*ray.dir_y-position.y;
        rayhit.hit.Ng_z = ray.org_z+t*ray.dir_z-position.z;
        FresnelRTCIntersectContext & context = *(FresnelRTCIntersectContext *)args->context;
        context.shading_color = geom->m_color->get(args->primID);

        // The distance of the hit position from the edge of the sphere,
        // projected into the plane which has the ray as it's normal
        const float d = radius - sqrt(Dsq);
        context.d = d;
        }
    }

/* Occlusion function, taken mostly from the embree user_geometry tutorial
 */
void GeometrySphere::occlude(const struct RTCOccludedFunctionNArguments *args)
    {
    GeometrySphere *geom = (GeometrySphere*)args->geometryUserPtr;
    const vec3<float> position = geom->m_position->get(args->primID);
    RTCRay& ray = *(RTCRay *)args->ray;
    const vec3<float> v = position-vec3<float>(ray.org_x,ray.org_y,ray.org_z);
    const float vsq = dot(v,v);
    const float radius = geom->m_radius->get(args->primID);
    const float rsq = (radius)*(radius);
    vec3<float> dir(ray.dir_x,ray.dir_y,ray.dir_z);
    const vec3<float> w = cross(v,dir);
    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(dir,dir);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(vsq - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    float t;
    if (dot(v, dir) > 0.0f)
        {
        if (vsq > rsq)
            {
            // ray origin is outside the sphere, compute the distance back from the closest point
            t = Rp-Ri;
            }
        else
            {
            // ray origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Rp+Ri;
            }
        }
    else
        {
        // origin is behind the sphere
        if (vsq > rsq)
            {
            // origin is outside the sphere, no intersection
            return;
            }
        else
            {
            // origin is inside the sphere, compute the distance to the outgoing intersection point
            t = Ri-Rp;
            }
        }

    if ((ray.tnear < t) & (t < ray.tfar))
        {
        ray.tfar = -std::numeric_limits<float>::infinity();
        }
    }

/*! \param m Python module to export in
 */
void export_GeometrySphere(pybind11::module& m)
    {
    pybind11::class_<GeometrySphere, std::shared_ptr<GeometrySphere> >(m, "GeometrySphere", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getPositionBuffer", &GeometrySphere::getPositionBuffer)
        .def("getRadiusBuffer", &GeometrySphere::getRadiusBuffer)
        .def("getColorBuffer", &GeometrySphere::getColorBuffer)
        ;
    }

} } // end namespace fresnel::cpu
