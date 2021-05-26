// Copyright (c) 2016-2021 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryPolygon.h"
#include "common/GeometryMath.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the polygon (in counterclockwise order)
    \param rounding_radius The rounding radius of the spheropolygon
    \param N number of primitives

    Initialize the polygon geometry.
*/
GeometryPolygon::GeometryPolygon(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
    float rounding_radius,
    unsigned int N)
    : Geometry(scene), m_rounding_radius(rounding_radius)
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
    m_position = std::shared_ptr<Array<vec2<float>>>(new Array<vec2<float>>(N));
    m_angle = std::shared_ptr<Array<float>>(new Array<float>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(N));

    // copy the vertices from the numpy array to internal storage
    pybind11::buffer_info info = vertices.request();

    if (info.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info.shape[1] != 2)
        throw std::runtime_error("vertices must be a Nvert by 2 array");

    float* verts_f = (float*)info.ptr;

    for (unsigned int i = 0; i < info.shape[0]; i++)
        {
        vec2<float> p0(verts_f[i * 2], verts_f[i * 2 + 1]);

        m_vertices.push_back(p0);

        // precompute radius in the xy plane
        m_radius = std::max(m_radius, sqrtf(dot(p0, p0)));
        }
    // pad the radius with the rounding radius
    m_radius += m_rounding_radius;

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryPolygon::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryPolygon::intersect);
    m_device->checkError();

    rtcCommitGeometry(m_geometry);
    m_device->checkError();

    m_valid = true;
    }

GeometryPolygon::~GeometryPolygon() { }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryPolygon instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryPolygon::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometryPolygon* geom = (GeometryPolygon*)args->geometryUserPtr;
    vec2<float> p2 = geom->m_position->get(args->primID);
    vec3<float> p(p2.x, p2.y, 0.0f);

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = p.x - geom->m_radius;
    bounds_o.lower_y = p.y - geom->m_radius;
    bounds_o.lower_z = p.z - 1e-5;

    bounds_o.upper_x = p.x + geom->m_radius;
    bounds_o.upper_y = p.y + geom->m_radius;
    bounds_o.upper_z = p.z + 1e-5;
    }

//! Test if a point is inside a polygon
/*! \param min_d  [out] minimum distance from p to the polygon edge
    \param p Point
    \param verts Polygon vertices

    \returns true if the point is inside the polygon

    \note \a p is *in the polygon's reference frame!*

    \ingroup overlap
*/
inline bool is_inside(float& min_d, const vec2<float>& p, const std::vector<vec2<float>>& verts)
    {
    // code for concave test from: http://alienryderflex.com/polygon/
    unsigned int nvert = verts.size();
    min_d = FLT_MAX;

    unsigned int i, j = nvert - 1;
    bool oddNodes = false;

    for (i = 0; i < nvert; i++)
        {
        min_d = fast::min(min_d, point_line_segment_distance(p, verts[i], verts[j]));

        if ((verts[i].y < p.y && verts[j].y >= p.y) || (verts[j].y < p.y && verts[i].y >= p.y))
            {
            if (verts[i].x
                    + (p.y - verts[i].y) / (verts[j].y - verts[i].y) * (verts[j].x - verts[i].x)
                < p.x)
                {
                oddNodes = !oddNodes;
                }
            }
        j = i;
        }

    return oddNodes;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryPolygon instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryPolygon::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    GeometryPolygon* geom = (GeometryPolygon*)args->geometryUserPtr;

    const vec2<float> p2 = geom->m_position->get(args->primID);
    const vec3<float> pos_world(p2.x, p2.y, 0.0f);
    const float angle = geom->m_angle->get(args->primID);
    const quat<float> q_world = quat<float>::fromAxisAngle(vec3<float>(0, 0, 1), angle);

    // transform the ray into the primitive coordinate system
    RTCRay& ray = ((RTCRayHit*)args->rayhit)->ray;
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z));
    vec3<float> ray_org_local
        = rotate(conj(q_world), vec3<float>(ray.org_x, ray.org_y, ray.org_z) - pos_world);

    // find the point where the ray intersects the plane of the polygon
    const vec3<float> n = vec3<float>(0, 0, 1);
    const vec3<float> p = vec3<float>(0, 0, 0);

    float d = -dot(n, p);
    float denom = dot(n, ray_dir_local);
    float t_hit = -(d + dot(n, ray_org_local)) / denom;

    // if the ray is parallel to the plane, there is no intersection
    if (fabs(denom) < 1e-5)
        {
        return;
        }

    // see if the intersection point is inside the polygon
    vec3<float> r_hit = ray_org_local + t_hit * ray_dir_local;
    vec2<float> r_hit_2d(r_hit.x, r_hit.y);
    float d_edge, min_d;

    bool inside = is_inside(d_edge, r_hit_2d, geom->m_vertices);

    // spheropolygon (equivalent to sharp polygon when rounding radius is 0
    // make distance signed (negative is inside)
    if (inside)
        {
        d_edge = -d_edge;
        }

    // exit if outside
    if (d_edge > geom->m_rounding_radius)
        {
        return;
        }
    min_d = geom->m_rounding_radius - d_edge;

    // if we get here, we hit the inside of the polygon
    // if the t_hit is in (tnear,tfar), we hit the polygon
    RTCRayHit& rh = *(RTCRayHit*)args->rayhit;
    FresnelRTCIntersectContext& context = *(FresnelRTCIntersectContext*)args->context;
    if ((ray.tnear < t_hit) & (t_hit < ray.tfar))
        {
        ray.tfar = t_hit;
        rh.hit.geomID = geom->m_geom_id;
        rh.hit.primID = args->primID;

        // make polygons double sided
        vec3<float> n_flip;
        if (dot(n, ray_dir_local) < 0.0f)
            {
            n_flip = n;
            }
        else
            {
            n_flip = -n;
            }

        vec3<float> Ng = rotate(q_world, n_flip);

        rh.hit.Ng_x = Ng.x;
        rh.hit.Ng_y = Ng.y;
        rh.hit.Ng_z = Ng.z;
        context.shading_color = geom->m_color->get(args->primID);
        context.d = min_d;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryPolygon(pybind11::module& m)
    {
    pybind11::class_<GeometryPolygon, Geometry, std::shared_ptr<GeometryPolygon>>(m,
                                                                                  "GeometryPolygon")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             float,
             unsigned int>())
        .def("getPositionBuffer", &GeometryPolygon::getPositionBuffer)
        .def("getAngleBuffer", &GeometryPolygon::getAngleBuffer)
        .def("getColorBuffer", &GeometryPolygon::getColorBuffer)
        .def("getRadius", &GeometryPolygon::getRadius);
    }

    } // namespace cpu
    } // namespace fresnel
