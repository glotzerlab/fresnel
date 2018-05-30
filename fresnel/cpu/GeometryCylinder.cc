// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>

#include "GeometryCylinder.h"
#include "common/IntersectCylinder.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param N number of cylinders to manage

    Initialize the cylinder geometry.
*/
GeometryCylinder::GeometryCylinder(std::shared_ptr<Scene> scene, unsigned int N)
    : Geometry(scene)
    {
    // create the geometry
    m_geom_id = rtcNewUserGeometry(m_scene->getRTCScene(), N);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1,0,1)));
    setOutlineMaterial(Material(RGB<float>(0,0,0), 1.0f));

    // initialize the buffers
    m_A = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_B = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_radius = std::shared_ptr< Array< float > >(new Array< float >(N));
    m_color_A = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));
    m_color_B = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));

    // register functions for embree
    rtcSetUserData(m_scene->getRTCScene(), m_geom_id, this);
    m_device->checkError();
    rtcSetBoundsFunction(m_scene->getRTCScene(), m_geom_id, &GeometryCylinder::bounds);
    m_device->checkError();
    rtcSetIntersectFunction(m_scene->getRTCScene(), m_geom_id, &GeometryCylinder::intersect);
    m_device->checkError();

    m_valid = true;
    }

GeometryCylinder::~GeometryCylinder()
    {
    }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryCylinder instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryCylinder::bounds(void *ptr, size_t item, RTCBounds& bounds_o)
    {
    GeometryCylinder *geom = (GeometryCylinder*)ptr;
    vec3<float> A = geom->m_A->get(item);
    vec3<float> B = geom->m_B->get(item);
    float radius = geom->m_radius->get(item);
    bounds_o.lower_x = std::min(A.x - radius, B.x - radius);
    bounds_o.lower_y = std::min(A.y - radius, B.y - radius);
    bounds_o.lower_z = std::min(A.z - radius, B.z - radius);

    bounds_o.upper_x = std::max(A.x + radius, B.x + radius);
    bounds_o.upper_y = std::max(A.y + radius, B.y + radius);
    bounds_o.upper_z = std::max(A.z + radius, B.z + radius);
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryCylinder instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryCylinder::intersect(void *ptr, RTCRay& ray, size_t item)
   {
    GeometryCylinder *geom = (GeometryCylinder*)ptr;
    const vec3<float> A = geom->m_A->get(item);
    const vec3<float> B = geom->m_B->get(item);
    const float radius = geom->m_radius->get(item);

    float t=HUGE_VALF, d=HUGE_VALF;
    vec3<float> N;
    unsigned int color_index;
    bool hit = intersect_ray_spherocylinder(t, d, N, color_index, ray.org, ray.dir, A, B, radius);

    if (hit && (ray.tnear < t) && (t < ray.tfar))
        {
        ray.u = 0.0f;
        ray.v = 0.0f;
        ray.tfar = t;
        ray.geomID = geom->m_geom_id;
        ray.primID = (unsigned int) item;
        ray.Ng = N;
        if (color_index == 0)
            ray.shading_color = geom->m_color_A->get(item);
        else
            ray.shading_color = geom->m_color_B->get(item);
        ray.d = d;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryCylinder(pybind11::module& m)
    {
    pybind11::class_<GeometryCylinder, std::shared_ptr<GeometryCylinder> >(m, "GeometryCylinder", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>, unsigned int>())
        .def("getABuffer", &GeometryCylinder::getABuffer)
        .def("getBBuffer", &GeometryCylinder::getBBuffer)
        .def("getRadiusBuffer", &GeometryCylinder::getRadiusBuffer)
        .def("getColorABuffer", &GeometryCylinder::getColorABuffer)
        .def("getColorBBuffer", &GeometryCylinder::getColorBBuffer)
        ;
    }

} } // end namespace fresnel::cpu
