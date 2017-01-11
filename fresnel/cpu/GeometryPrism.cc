// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryPrism.h"

namespace fresnel { namespace cpu {

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
    std::cout << "Create GeometryPrism" << std::endl;
    // create the geometry
    m_geom_id = rtcNewUserGeometry(m_scene->getRTCScene(), N);
    m_device->checkError();

    // allocate buffer data
    m_position = std::shared_ptr< Array< vec2<float> > >(new Array< vec2<float> >(N));
    m_angle = std::shared_ptr< Array< float > >(new Array< float >(N));
    m_height = std::shared_ptr< Array< float > >(new Array< float >(N));
    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));

    // set up the planes. The first two planes are the top (z=height) and bottom (z=0).
    // initialize both to 0 here, and other code will set the height appropriately
    m_plane_origin.push_back(vec3<float>(0,0,0));
    m_plane_normal.push_back(vec3<float>(0,0,1));
    m_plane_origin.push_back(vec3<float>(0,0,0));
    m_plane_normal.push_back(vec3<float>(0,0,-1));

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

        m_plane_origin.push_back(vec3<float>(p0.x, p0.y, 0));
        m_plane_normal.push_back(vec3<float>(n.x, n.y, 0));

        // validate winding order
        int k = (j + 1) % info.shape[0];
        vec2<float> p2(verts_f[k*2], verts_f[k*2+1]);

        if (perpdot(p1 - p0, p2 - p1) <= 0)
            throw std::invalid_argument("vertices must be counterclockwise and convex");

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
    vec2<float> p2 = geom->m_position->get(item);
    float height = geom->m_height->get(item);
    vec3<float> p(p2.x, p2.y, 0.0f);

    bounds_o.lower_x = p.x - geom->m_radius;
    bounds_o.lower_y = p.y - geom->m_radius;
    bounds_o.lower_z = p.z;

    bounds_o.upper_x = p.x + geom->m_radius;
    bounds_o.upper_y = p.y + geom->m_radius;
    bounds_o.upper_z = p.z + height;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryPrism instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryPrism::intersect(void *ptr, RTCRay& ray, size_t item)
    {
    GeometryPrism *geom = (GeometryPrism*)ptr;

    // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
    int n_planes = geom->m_plane_normal.size();
    float t0 = -std::numeric_limits<float>::max();
    float t1 = std::numeric_limits<float>::max();

    const vec2<float> p2 = geom->m_position->get(item);
    const vec3<float> pos_world(p2.x, p2.y, 0.0f);
    const float angle = geom->m_angle->get(item);
    const float height = geom->m_height->get(item);
    const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0,0,1), angle);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), ray.dir);
    vec3<float> ray_org_local = rotate(conj(q_world), ray.org - pos_world);

    vec3<float> t0_n_local, t0_p_local;
    vec3<float> t1_n_local, t1_p_local;
    for(int i = 0; i < n_planes && t0 < t1; ++i )
        {
        vec3<float> n = geom->m_plane_normal[i];
        vec3<float> p = geom->m_plane_origin[i];

        // correct the top plane positions
        if (i == 0)
            p.z = height;

        float d = -dot(n, p);
        float denom = dot(n, ray_dir_local);
        float t = -(d + dot(n, ray_org_local))/denom;

        // if the ray is parallel to the plane, there is no intersection when the ray is outside the shape
        if (fabs(denom) < 1e-5)
            {
            if (dot(ray_org_local - p, n) > 0)
                return;
            }
        else if (denom < 0)
            {
            // find the last plane this ray enters
            if(t > t0)
                {
                t0 = t;
                t0_n_local = n;
                t0_p_local = p;
                }
            }
        else
            {
            // find the first plane this ray exits
            if(t < t1)
                {
                t1 = t;
                t1_n_local = n;
                t1_p_local = p;
                }
            }
        }

    // if the ray enters after it exits, it missed the polyhedron
    if(t0 > t1)
        return;

    // otherwise, it hit: fill out the hit structure and track the plane that was hit
    float t_hit = 0;
    bool hit = false;
    vec3<float> n_hit, p_hit;

    // if the t0 is in (tnear,tfar), we hit the entry plane
    if ((ray.tnear < t0) & (t0 < ray.tfar))
        {
        t_hit = ray.tfar = t0;
        ray.geomID = geom->m_geom_id;
        ray.primID = item;
        ray.Ng = rotate(q_world, t0_n_local);
        n_hit = t0_n_local;
        p_hit = t0_p_local;
        ray.shading_color = geom->m_color->get(item);
        hit = true;
        }
    // if t1 is in (tnear,tfar), we hit the exit plane
    if ((ray.tnear < t1) & (t1 < ray.tfar))
        {
        t_hit = ray.tfar = t1;
        ray.geomID = geom->m_geom_id;
        ray.primID = item;
        ray.Ng = rotate(q_world, t1_n_local);
        n_hit = t1_n_local;
        p_hit = t1_p_local;
        ray.shading_color = geom->m_color->get(item);
        hit = true;
        }

    // determine distance from the hit point to the nearest edge
    float min_d = std::numeric_limits<float>::max();
    vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;
    if (hit)
        {
        // edges come from intersections of planes
        // loop over all planes and find the intersection with the hit plane
        for(int i = 0; i < n_planes; ++i )
            {
            vec3<float> n = geom->m_plane_normal[i];
            vec3<float> p = geom->m_plane_origin[i];

            // correct the top plane positions
            if (i == 0)
                p.z = geom->m_height->get(item);

            // ********
            // find the line of intersection between the two planes
            // adapted from: http://geomalgorithms.com/a05-_intersect-1.html

            // direction of the line
            vec3<float> u = cross(n, n_hit);

            // if the planes are not coplanar
            if (fabs(dot(u,u)) >= 1e-5)
                {
                int maxc; // max coordinate
                if (fabs(u.x) > fabs(u.y))
                    {
                    if (fabs(u.x) > fabs(u.z))
                        maxc = 1;
                    else
                        maxc = 3;
                    }
                else
                    {
                    if (fabs(u.y) > fabs(u.z))
                        maxc = 2;
                    else
                        maxc = 3;
                    }

                // a point on the line
                vec3<float> x0;
                float d1 = -dot(n,p);
                float d2 = -dot(n_hit, p_hit);

                // solve the problem in different ways based on which direction is maximum
                switch (maxc)
                    {
                    case 1:                     // intersect with x=0
                        x0.x = 0;
                        x0.y = (d2*n.z - d1*n_hit.z) /  u.x;
                        x0.z = (d1*n_hit.y - d2*n.y) /  u.x;
                        break;
                    case 2:                     // intersect with y=0
                        x0.x = (d1*n_hit.z - d2*n.z) /  u.y;
                        x0.y = 0;
                        x0.z = (d2*n.x - d1*n_hit.x) /  u.y;
                        break;
                    case 3:                     // intersect with z=0
                        x0.x = (d2*n.y - d1*n_hit.y) /  u.z;
                        x0.y = (d1*n_hit.x - d2*n.x) /  u.z;
                        x0.z = 0;
                    }

                // we want the distance in the view plane for consistent line edge widths
                // project the line x0 + t*u into the plane perpendicular to the view direction passing through r_hit
                vec3<float> view = -ray_dir_local / sqrtf(dot(ray_dir_local, ray_dir_local));
                u = u - dot(u, view) * view;
                vec3<float> w = x0 - r_hit;
                vec3<float> w_perp = w - dot(w, view) * view;
                x0 = r_hit + w_perp;

                // ********
                // find the distance from the hit point to the line
                // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                vec3<float> v = cross(u, x0 - r_hit);
                float dsq = dot(v, v) / dot(u,u);
                float d = sqrtf(dsq);
                if (d < min_d)
                    min_d = d;
                }
            }
        ray.d = min_d;
        }
    }

/*! Test if a ray intersects with the given primitive

    \param ptr Pointer to a GeometryPrism instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryPrism::occlude(void *ptr, RTCRay& ray, size_t item)
    {
    // this method is a copy and pasted version of intersect with a different behavior on hit, to
    // meet Embree API standards. When intersect is updated, it should be copied and pasted back here.
    GeometryPrism *geom = (GeometryPrism*)ptr;

    // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
    int n_planes = geom->m_plane_normal.size();
    float t0 = -std::numeric_limits<float>::max();
    float t1 = std::numeric_limits<float>::max();

    const vec2<float> p2 = geom->m_position->get(item);
    const vec3<float> pos_world(p2.x, p2.y, 0.0f);
    const float angle = geom->m_angle->get(item);
    const float height = geom->m_height->get(item);
    const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0,0,1), angle);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), ray.dir);
    vec3<float> ray_org_local = rotate(conj(q_world), ray.org - pos_world);

    vec3<float> t0_n_local, t0_p_local;
    vec3<float> t1_n_local, t1_p_local;
    for(int i = 0; i < n_planes && t0 < t1; ++i )
        {
        vec3<float> n = geom->m_plane_normal[i];
        vec3<float> p = geom->m_plane_origin[i];

        // correct the top plane positions
        if (i == 0)
            p.z = height;

        float d = -dot(n, p);
        float denom = dot(n, ray_dir_local);
        float t = -(d + dot(n, ray_org_local))/denom;

        // if the ray is parallel to the plane, there is no intersection when the ray is outside the shape
        if (fabs(denom) < 1e-5)
            {
            if (dot(ray_org_local - p, n) > 0)
                return;
            }
        else if (denom < 0)
            {
            // find the last plane this ray enters
            if(t > t0)
                {
                t0 = t;
                t0_n_local = n;
                }
            }
        else
            {
            // find the first plane this ray exits
            if(t < t1)
                {
                t1 = t;
                t1_n_local = n;
                }
            }
        }

    // if the ray enters after it exits, it missed the polyhedron
    if(t0 > t1)
        return;

    // otherwise, it hit: fill out the hit structure

    // if the t0 is in (tnear,tfar), we hit the entry plane
    if ((ray.tnear < t0) & (t0 < ray.tfar))
        {
        ray.geomID = 0;
        }
    // if t1 is in (tnear,tfar), we hit the exit plane
    if ((ray.tnear < t1) & (t1 < ray.tfar))
        {
        ray.geomID = 0;
        }
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
        ;
    }

} } // end namespace fresnel::cpu
