// Copyright (c) 2016-2021 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryMesh.h"
#include "common/IntersectTriangle.h"

namespace fresnel
    {
namespace cpu
    {
/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the mesh
    \param N Number of polyhedra
    Initialize the mesh.
*/
GeometryMesh::GeometryMesh(
    std::shared_ptr<Scene> scene,
    pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
    unsigned int N)
    : Geometry(scene)
    {
    // extract vertices array from numpy
    pybind11::buffer_info info_vertices = vertices.request();

    if (info_vertices.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nvert by 3 array");

    if (info_vertices.shape[0] % 3 != 0)
        throw std::runtime_error("the number of triangle vertices must be a multiple of three.");

    unsigned int n_faces = info_vertices.shape[0] / 3;
    unsigned int n_verts = info_vertices.shape[0];
    float* verts_f = (float*)info_vertices.ptr;

    // allocate buffer data
    m_position = std::shared_ptr<Array<vec3<float>>>(new Array<vec3<float>>(N));
    m_orientation = std::shared_ptr<Array<quat<float>>>(new Array<quat<float>>(N));
    m_color = std::shared_ptr<Array<RGB<float>>>(new Array<RGB<float>>(n_verts));

    // copy vertices into local buffer
    m_vertices.resize(n_verts);
    memcpy((void*)&m_vertices[0], verts_f, sizeof(vec3<float>) * n_verts);

    // create the geometry
    m_geometry = rtcNewGeometry(m_device->getRTCDevice(), RTC_GEOMETRY_TYPE_USER);
    m_device->checkError();
    rtcSetGeometryUserPrimitiveCount(m_geometry, N * n_faces);
    m_device->checkError();
    m_geom_id = rtcAttachGeometry(m_scene->getRTCScene(), m_geometry);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1, 0, 1)));
    setOutlineMaterial(Material(RGB<float>(0, 0, 0), 1.0f));

    // register functions for embree
    rtcSetGeometryUserData(m_geometry, this);
    m_device->checkError();
    rtcSetGeometryBoundsFunction(m_geometry, &GeometryMesh::bounds, NULL);
    m_device->checkError();
    rtcSetGeometryIntersectFunction(m_geometry, &GeometryMesh::intersect);
    m_device->checkError();

    rtcCommitGeometry(m_geometry);
    m_device->checkError();

    m_valid = true;
    }

GeometryMesh::~GeometryMesh() { }

/*! Compute the bounding box of a given primitive

    \param args Arguments to the bounds check
*/
void GeometryMesh::bounds(const struct RTCBoundsFunctionArguments* args)
    {
    GeometryMesh* geom = (GeometryMesh*)args->geometryUserPtr;

    unsigned int item = args->primID;
    unsigned int n_faces = geom->m_vertices.size() / 3;
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    vec3<float> p3 = geom->m_position->get(i_poly);
    quat<float> q_world = geom->m_orientation->get(i_poly);

    // rotate vertices into space frame
    const vec3<float> v0(geom->m_vertices[i_face * 3]);
    const vec3<float> v1(geom->m_vertices[i_face * 3 + 1]);
    const vec3<float> v2(geom->m_vertices[i_face * 3 + 2]);

    vec3<float> v0_world = rotate(q_world, v0) + p3;
    vec3<float> v1_world = rotate(q_world, v1) + p3;
    vec3<float> v2_world = rotate(q_world, v2) + p3;

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = std::min(v0_world.x, std::min(v1_world.x, v2_world.x));
    bounds_o.lower_y = std::min(v0_world.y, std::min(v1_world.y, v2_world.y));
    bounds_o.lower_z = std::min(v0_world.z, std::min(v1_world.z, v2_world.z));

    bounds_o.upper_x = std::max(v0_world.x, std::max(v1_world.x, v2_world.x));
    bounds_o.upper_y = std::max(v0_world.y, std::max(v1_world.y, v2_world.y));
    bounds_o.upper_z = std::max(v0_world.z, std::max(v1_world.z, v2_world.z));
    }

/*! Compute the intersection of a ray with the given primitive
   \param args Arguments to the intersect check
*/
void GeometryMesh::intersect(const struct RTCIntersectFunctionNArguments* args)
    {
    GeometryMesh* geom = (GeometryMesh*)args->geometryUserPtr;

    unsigned int item = args->primID;
    unsigned int n_faces = geom->m_vertices.size() / 3;
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    RTCRayHit& rayhit = *(RTCRayHit*)args->rayhit;
    RTCRay& ray = rayhit.ray;

    const vec3<float> p3 = geom->m_position->get(i_poly);
    const quat<float> q_world = geom->m_orientation->get(i_poly);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.dir_x, ray.dir_y, ray.dir_z));
    vec3<float> ray_org_local
        = rotate(conj(q_world), vec3<float>(ray.org_x, ray.org_y, ray.org_z) - p3);

    vec3<float> v0 = geom->m_vertices[i_face * 3];
    vec3<float> v1 = geom->m_vertices[i_face * 3 + 1];
    vec3<float> v2 = geom->m_vertices[i_face * 3 + 2];
    float u, v, w, t, d;
    vec3<float> n;

    // double-sided triangle test
    if (!intersect_ray_triangle(u,
                                v,
                                w,
                                t,
                                d,
                                n,
                                ray_org_local,
                                ray_org_local + ray_dir_local,
                                v0,
                                v1,
                                v2)
        && !intersect_ray_triangle(v,
                                   u,
                                   w,
                                   t,
                                   d,
                                   n,
                                   ray_org_local,
                                   ray_org_local + ray_dir_local,
                                   v1,
                                   v0,
                                   v2))
        return;

    // if the t is in (tnear,tfar), we hit the entry plane
    if ((ray.tnear < t) & (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = item;
        vec3<float> n_world = rotate(q_world, n);

        rayhit.hit.Ng_x = n_world.x;
        rayhit.hit.Ng_y = n_world.y;
        rayhit.hit.Ng_z = n_world.z;

        FresnelRTCIntersectContext& context = *(FresnelRTCIntersectContext*)args->context;
        rayhit.hit.instID[0] = context.context.instID[0];

        context.shading_color = geom->m_color->get(i_face * 3 + 0) * u
                                + geom->m_color->get(i_face * 3 + 1) * v
                                + geom->m_color->get(i_face * 3 + 2) * w;

        context.d = d;
        }
    }

/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, Geometry, std::shared_ptr<GeometryMesh>>(m, "GeometryMesh")
        .def(pybind11::init<
             std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryMesh::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryMesh::getOrientationBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer);
    }

    } // namespace cpu
    } // namespace fresnel
