// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryConvexPolyhedron.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param plane_origins Origins of the planes that make up the polyhedron
    \param plane_normals Normals of the planes that make up the polyhedron
    \param r radius of the polyhedron

    Initialize the polyhedron geometry.
*/
GeometryConvexPolyhedron::GeometryConvexPolyhedron(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_origins,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_normals,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> plane_colors,
    unsigned int N,
    float r)
    : Geometry(scene)
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

    // allocate buffer data
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // access the plane data
    pybind11::buffer_info info_origin = plane_origins.request();

    if (info_origin.ndim != 2)
        throw std::runtime_error("plane_origins must be a 2-dimensional array");

    if (info_origin.shape[1] != 3)
        throw std::runtime_error("plane_origins must be a Nvert by 3 array");

    float* origin_f = (float*)info_origin.ptr;

    pybind11::buffer_info info_normal = plane_normals.request();

    if (info_normal.ndim != 2)
        throw std::runtime_error("plane_normals must be a 2-dimensional array");

    if (info_normal.shape[1] != 3)
        throw std::runtime_error("plane_normals must be a Nvert by 3 array");

    if (info_normal.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and normal arrays");

    float* normal_f = (float*)info_normal.ptr;

    pybind11::buffer_info info_color = plane_colors.request();

    if (info_color.ndim != 2)
        throw std::runtime_error("plane_colors must be a 2-dimensional array");

    if (info_color.shape[1] != 3)
        throw std::runtime_error("plane_colors must be a Nvert by 3 array");

    if (info_color.shape[0] != info_origin.shape[0])
        throw std::runtime_error("Number of vertices must match in origin and color arrays");

    float* color_f = (float*)info_color.ptr;

    // construct planes in C++ data structures
    for (unsigned int i = 0; i < info_normal.shape[0]; i++)
        {
        vec3<float> n(normal_f[i * 3], normal_f[i * 3 + 1], normal_f[i * 3 + 2]);
        n = n / sqrtf(dot(n, n));

        m_plane_origin.push_back(
            vec3<float>(origin_f[i * 3], origin_f[i * 3 + 1], origin_f[i * 3 + 2]));
        m_plane_normal.push_back(vec3<float>(n.x, n.y, n.z));
        m_plane_color.push_back(RGB<float>(color_f[i * 3], color_f[i * 3 + 1], color_f[i * 3 + 2]));
        }

    // for now, take a user supplied radius
    m_radius = r;

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryConvexPolyhedron::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryConvexPolyhedron::intersect);
    m_device->checkError();

    rtcCommitGeometry(m_geometry);
    m_device->checkError();

    m_valid = true;
    }

GeometryConvexPolyhedron::~GeometryConvexPolyhedron() { }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryConvexPolyhedron instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryConvexPolyhedron::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometryConvexPolyhedron* geom = (GeometryConvexPolyhedron*)args->geometryUserPtr;
    vec3<float> p = geom->m_position->get(args->primID);

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = p.x - geom->m_radius;
    bounds_o.lower_y = p.y - geom->m_radius;
    bounds_o.lower_z = p.z - geom->m_radius;

    bounds_o.upper_x = p.x + geom->m_radius;
    bounds_o.upper_y = p.y + geom->m_radius;
    bounds_o.upper_z = p.z + geom->m_radius;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryConvexPolyhedron instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryConvexPolyhedron::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    *args->valid = 0;

    GeometryConvexPolyhedron* geom = (GeometryConvexPolyhedron*)args->geometryUserPtr;

    // adapted from OptiX quick start tutorial and Embree user_geometry tutorial files
    int n_planes = geom->m_plane_normal.size();
    float t0 = -std::numeric_limits<float>::max();
    float t1 = std::numeric_limits<float>::max();

    const vec3<float> pos_world = geom->m_position->get(args->primID);
    const quat<float> q_world = geom->m_orientation->get(args->primID);

    RTCRayHit& rayhit = *(RTCRayHit*)args->rayhit;
    RTCRay& ray = rayhit.ray;
    vec3<float> dir = vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), dir);
    vec3<float> ray_org_local
        = rotate(conj(q_world), vec3<float>(ray.org_x, ray.org_y, ray.org_z) - pos_world);

    vec3<float> t0_n_local(0, 0, 0), t0_p_local(0, 0, 0);
    vec3<float> t1_n_local(0, 0, 0), t1_p_local(0, 0, 0);
    int t0_plane_hit = 0, t1_plane_hit = 0;
    for (int i = 0; i < n_planes && t0 < t1; ++i)
        {
        vec3<float> n = geom->m_plane_normal[i];
        vec3<float> p = geom->m_plane_origin[i];

        float d = -dot(n, p);
        float denom = dot(n, ray_dir_local);
        float t = -(d + dot(n, ray_org_local)) / denom;

        // if the ray is parallel to the plane, there is no intersection when the ray is outside the
        // shape
        if (fabs(denom) < 1e-5)
            {
            if (dot(ray_org_local - p, n) > 0)
                return;
            }
        else if (denom < 0)
            {
            // find the last plane this ray enters
            if (t > t0)
                {
                t0 = t;
                t0_n_local = n;
                t0_p_local = p;
                t0_plane_hit = i;
                }
            }
        else
            {
            // find the first plane this ray exits
            if (t < t1)
                {
                t1 = t;
                t1_n_local = n;
                t1_p_local = p;
                t1_plane_hit = i;
                }
            }
        }

    // if the ray enters after it exits, it missed the polyhedron
    if (t0 > t1)
        return;

    // otherwise, it hit: fill out the hit structure and track the plane that was hit
    float t_hit = 0;
    bool hit = false;
    vec3<float> n_hit, p_hit;

    // if the t0 is in (tnear,tfar), we hit the entry plane
    FresnelRTCIntersectContext& context = *(FresnelRTCIntersectContext*)args->context;
    if ((ray.tnear < t0) & (t0 < ray.tfar))
        {
        t_hit = ray.tfar = t0;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = args->primID;
        vec3<float> Ng = rotate(q_world, t0_n_local);
        rayhit.hit.Ng_x = Ng.x;
        rayhit.hit.Ng_y = Ng.y;
        rayhit.hit.Ng_z = Ng.z;
        n_hit = t0_n_local;
        p_hit = t0_p_local;
        context.shading_color = lerp(geom->m_color_by_face,
                                     geom->m_color->get(args->primID),
                                     geom->m_plane_color[t0_plane_hit]);
        hit = true;
        }
    // if t1 is in (tnear,tfar), we hit the exit plane
    if ((ray.tnear < t1) & (t1 < ray.tfar))
        {
        t_hit = ray.tfar = t1;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = args->primID;
        vec3<float> Ng = rotate(q_world, t1_n_local);
        rayhit.hit.Ng_x = Ng.x;
        rayhit.hit.Ng_y = Ng.y;
        rayhit.hit.Ng_z = Ng.z;
        n_hit = t1_n_local;
        p_hit = t1_p_local;
        context.shading_color = lerp(geom->m_color_by_face,
                                     geom->m_color->get(args->primID),
                                     geom->m_plane_color[t1_plane_hit]);
        hit = true;
        }

    // determine distance from the hit point to the nearest edge
    float min_d = std::numeric_limits<float>::max();
    vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;
    if (hit)
        {
        // edges come from intersections of planes
        // loop over all planes and find the intersection with the hit plane
        for (int i = 0; i < n_planes; ++i)
            {
            vec3<float> n = geom->m_plane_normal[i];
            vec3<float> p = geom->m_plane_origin[i];

            // ********
            // find the line of intersection between the two planes
            // adapted from: http://geomalgorithms.com/a05-_intersect-1.html

            // direction of the line
            vec3<float> u = cross(n, n_hit);

            // if the planes are not coplanar
            if (fabs(dot(u, u)) >= 1e-5)
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
                float d1 = -dot(n, p);
                float d2 = -dot(n_hit, p_hit);

                // solve the problem in different ways based on which direction is maximum
                switch (maxc)
                    {
                case 1: // intersect with x=0
                    x0.x = 0;
                    x0.y = (d2 * n.z - d1 * n_hit.z) / u.x;
                    x0.z = (d1 * n_hit.y - d2 * n.y) / u.x;
                    break;
                case 2: // intersect with y=0
                    x0.x = (d1 * n_hit.z - d2 * n.z) / u.y;
                    x0.y = 0;
                    x0.z = (d2 * n.x - d1 * n_hit.x) / u.y;
                    break;
                case 3: // intersect with z=0
                    x0.x = (d2 * n.y - d1 * n_hit.y) / u.z;
                    x0.y = (d1 * n_hit.x - d2 * n.x) / u.z;
                    x0.z = 0;
                    }

                // we want the distance in the view plane for consistent line edge widths
                // project the line x0 + t*u into the plane perpendicular to the view direction
                // passing through r_hit
                vec3<float> view = -ray_dir_local / sqrtf(dot(ray_dir_local, ray_dir_local));
                u = u - dot(u, view) * view;
                vec3<float> w = x0 - r_hit;
                vec3<float> w_perp = w - dot(w, view) * view;
                x0 = r_hit + w_perp;

                // ********
                // find the distance from the hit point to the line
                // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                vec3<float> v = cross(u, x0 - r_hit);
                float dsq = dot(v, v) / dot(u, u);
                float d = sqrtf(dsq);
                if (d < min_d)
                    min_d = d;
                }
            }
        context.d = min_d;
        *args->valid = -1;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryConvexPolyhedron(pybind11::module& m)
    {
    pybind11::class_<GeometryConvexPolyhedron, Geometry, std::shared_ptr<GeometryConvexPolyhedron>>(
        m,
        "GeometryConvexPolyhedron")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int,
             float>())
        .def("getPositionBuffer", &GeometryConvexPolyhedron::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryConvexPolyhedron::getOrientationBuffer)
        .def("getColorBuffer", &GeometryConvexPolyhedron::getColorBuffer)
        .def("setColorByFace", &GeometryConvexPolyhedron::setColorByFace)
        .def("getColorByFace", &GeometryConvexPolyhedron::getColorByFace);
    }

    } // namespace cpu
    } // namespace fresnel
