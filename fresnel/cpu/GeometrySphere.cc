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
    std::cout << "Create GeometrySphere" << std::endl;
    // create the geometry
    m_geom_id = rtcNewUserGeometry(m_scene->getRTCScene(), N);
    m_device->checkError();

    // initialize the buffers
    m_position = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_radius = std::shared_ptr< Array< float > >(new Array< float >(N));
    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));

    // register functions for embree
    rtcSetUserData(m_scene->getRTCScene(), m_geom_id, this);
    m_device->checkError();
    rtcSetBoundsFunction(m_scene->getRTCScene(), m_geom_id, &GeometrySphere::bounds);
    m_device->checkError();
    rtcSetIntersectFunction(m_scene->getRTCScene(), m_geom_id, &GeometrySphere::intersect);
    m_device->checkError();
    rtcSetOccludedFunction(m_scene->getRTCScene(), m_geom_id, &GeometrySphere::occlude);
    m_device->checkError();

    m_valid = true;
    }

GeometrySphere::~GeometrySphere()
    {
    std::cout << "Destroy GeometrySphere" << std::endl;
    }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometrySphere instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometrySphere::bounds(void *ptr, size_t item, RTCBounds& bounds_o)
    {
    GeometrySphere *geom = (GeometrySphere*)ptr;
    vec3<float> p = geom->m_position->get(item);
    float radius = geom->m_radius->get(item);
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
void GeometrySphere::intersect(void *ptr, RTCRay& ray, size_t item)
   {
    GeometrySphere *geom = (GeometrySphere*)ptr;
    const vec3<float> position = geom->m_position->get(item);
    const vec3<float> v = ray.org-position;
    const float radius = geom->m_radius->get(item);
    const float rsq = (radius)*(radius);
    const vec3<float> w = cross(ray.dir,v);
    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(ray.dir,ray.dir);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(dot(v,v) - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    const float t0 = Rp-Ri;
    const float t1 = Rp+Ri;

    if ((ray.tnear < t0) & (t0 < ray.tfar))
        {
        ray.u = 0.0f;
        ray.v = 0.0f;
        ray.tfar = t0;
        ray.geomID = geom->m_geom_id;
        ray.primID = (unsigned int) item;
        ray.Ng = ray.org+t0*ray.dir-position;
        ray.shading_color = geom->m_color->get(item);
        }
    if ((ray.tnear < t1) & (t1 < ray.tfar))
        {
        ray.u = 0.0f;
        ray.v = 0.0f;
        ray.tfar = t1;
        ray.geomID = geom->m_geom_id;
        ray.primID = (unsigned int) item;
        ray.Ng = ray.org+t1*ray.dir-position;
        ray.shading_color = geom->m_color->get(item);
        }

    // The distance of the hit position from the edge of the sphere,
    // projected into the plane which has the ray as it's normal
    const float d = radius - sqrt(Dsq);
    if (d < ray.d)
        ray.d = d;
    }

/* Occlusion function, taken mostly from the embree user_geometry tutorial
 */
void GeometrySphere::occlude(void *ptr, RTCRay& ray, size_t item)
   {
    GeometrySphere *geom = (GeometrySphere*)ptr;
    const vec3<float> position = geom->m_position->get(item);
    const vec3<float> v = ray.org-position;
    const float radius = geom->m_radius->get(item);
    const float rsq = (radius)*(radius);
    const vec3<float> w = cross(ray.dir,v);
    // Closest point-line distance, taken from
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    const float Dsq = dot(w,w)/dot(ray.dir,ray.dir);
    if (Dsq > rsq) return; // a miss
    const float Rp = sqrt(dot(v,v) - Dsq); //Distance to closest point
    //Distance from clostest point to point on sphere
    const float Ri = sqrt(rsq - Dsq);
    const float t0 = Rp-Ri;
    const float t1 = Rp+Ri;

    if ((ray.tnear < t0) & (t0 < ray.tfar))
        {
        ray.geomID = 0;
        }
    if ((ray.tnear < t1) & (t1 < ray.tfar))
        {
        ray.geomID = 0;
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
