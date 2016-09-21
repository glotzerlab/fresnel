// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>

#include "GeometryPrism.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the polygon (in counterclockwise order)
    \param position position of each prism
    \param orientation orientation angle of each prism
    \param height height of each prism

    Creates a triangle mesh geometry with the given vertices and indices.
*/
GeometryPrism::GeometryPrism(std::shared_ptr<Scene> scene,
                             const std::vector<std::tuple<float, float> > &vertices,
                             const std::vector<std::tuple<float, float> > &position,
                             const std::vector< float > &orientation,
                             const std::vector< float > &height)
    : Geometry(scene)
    {
    std::cout << "Create GeometryPrism" << std::endl;
    // create the geometry
    m_geom_id = rtcNewUserGeometry(m_scene->getRTCScene(), position.size());
    m_device->checkError();

    // copy data into the local buffers
    m_position.resize(position.size());
    if (orientation.size() != position.size())
        throw std::invalid_argument("orientation must have the same length as position");
    m_orientation.resize(orientation.size());
    if (height.size() != position.size())
        throw std::invalid_argument("height must have the same length as position");
    m_height = height;

    for (unsigned int i = 0; i < position.size(); i++)
        {
        m_position[i] = vec3<float>(std::get<0>(position[i]), std::get<0>(position[i]), 0);
        m_orientation[i] = quat<float>::fromAxisAngle(vec3<float>(0,0,1), orientation[i]);
        }

    // set up the planes. The first two planes are the top (z=height) and bottom (z=0).
    // initialize both to 0 here, and other code will set the height appropriately
    m_plane_origin.push_back(vec3<float>(0,0,0));
    m_plane_normal.push_back(vec3<float>(0,0,1));
    m_plane_origin.push_back(vec3<float>(0,0,0));
    m_plane_normal.push_back(vec3<float>(0,0,-1));

    // now create planes for each of the polygon edges
    for (unsigned int i = 0; i < vertices.size(); i++)
        {
        vec2<float> p0(std::get<0>(vertices[i]), std::get<1>(vertices[i]));
        int j = (i + 1) % vertices.size();
        vec2<float> p1(std::get<0>(vertices[j]), std::get<1>(vertices[j]));
        vec2<float> n = -perp(p1 - p0);

        // TODO: validate winding order

        m_plane_origin.push_back(vec3<float>(p0.x, p0.y, 0));
        m_plane_normal.push_back(vec3<float>(n.x, n.y, 0));

        // precompute radius in the xy plane
        m_radius = std::max(m_radius, sqrtf(dot(p0,p0)));
        }

    // register functions for embree
    rtcSetUserData(m_scene->getRTCScene(), m_geom_id, this);
    m_device->checkError();
    rtcSetBoundsFunction(m_scene->getRTCScene(), m_geom_id, &GeometryPrism::bounds);
    m_device->checkError();
    rtcSetIntersectFunction(m_scene->getRTCScene(), m_geom_id, &GeometryPrism::intersect);
    m_device->checkError();
    rtcSetOccludedFunction(m_scene->getRTCScene(), m_geom_id, &GeometryPrism::occlude);
    m_device->checkError();

    m_valid = true;
    }

GeometryPrism::~GeometryPrism()
    {
    std::cout << "Destroy GeometryPrism" << std::endl;
    }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryPrism instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryPrism::bounds(void *ptr, size_t item, RTCBounds& bounds_o)
    {
    GeometryPrism *geom = (GeometryPrism*)ptr;
    vec3<float> p = geom->m_position[item];
    bounds_o.lower_x = p.x - geom->m_radius;
    bounds_o.lower_y = p.y - geom->m_radius;
    bounds_o.lower_z = p.z;

    bounds_o.upper_x = p.x + geom->m_radius;
    bounds_o.upper_y = p.y + geom->m_radius;
    bounds_o.upper_z = p.z + geom->m_height[item];
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryPrism instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryPrism::intersect(void *ptr, RTCRay& ray, size_t item)
    {
    GeometryPrism *geom = (GeometryPrism*)ptr;
    // TODO: implement
    ray.geomID = geom->m_geom_id;
    }

/*! Test if a ray intersects with the given primitive

    \param ptr Pointer to a GeometryPrism instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryPrism::occlude(void *ptr, RTCRay& ray, size_t item)
    {
    // GeometryPrism *geom = (GeometryPrism*)ptr;
    // TODO: implement
    ray.geomID = 0;
    }

/*! \param m Python module to export in
 */
void export_GeometryPrism(pybind11::module& m)
    {
    pybind11::class_<GeometryPrism, std::shared_ptr<GeometryPrism> >(m, "GeometryPrism", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>,
             const std::vector<std::tuple<float, float> > &,
             const std::vector<std::tuple<float, float> > &,
             const std::vector< float > &,
             const std::vector< float > &
             >())
        ;
    }

} } // end namespace fresnel::cpu
