// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryMesh.h"

namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the mesh
    \param triangles triangle indices of the mesh
    \param N Number of polyhedra
    Initialize the mesh.
*/
GeometryMesh::GeometryMesh(std::shared_ptr<Scene> scene,
                             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                             pybind11::array_t<unsigned int, pybind11::array::c_style | pybind11::array::forcecast> triangles,
                             unsigned int N)
    : Geometry(scene)
    {
    // allocate buffer data
    m_position = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_orientation = std::shared_ptr< Array< quat<float>  > >(new Array< quat<float> >(N));
    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(N));

    // now create planes for each of the polygon edges
    pybind11::buffer_info info_vertices = vertices.request();

    if (info_vertices.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nvert by 3 array");

    pybind11::buffer_info info_faces = triangles.request();
    if (info_faces.ndim != 2)
        throw std::runtime_error("faces must be a 2-dimensional array");

    if (info_faces.shape[1] != 3)
        throw std::runtime_error("faces must be a Nvert by 3 array");

    unsigned int n_faces = info_faces.shape[0];
    unsigned int n_verts = info_vertices.shape[0];

    float *verts_f = (float *)info_vertices.ptr;
    unsigned int *faces_f = (unsigned int *)info_faces.ptr;

    m_radius.resize(n_faces,0.0);
    m_face_origin.resize(n_faces);

    for (unsigned int i = 0; i < n_faces; i++)
        {
        // construct the normal and origin of each plane
        unsigned int t1 = faces_f[3*i];
        unsigned int t2 = faces_f[3*i+1];
        unsigned int t3 = faces_f[3*i+2];

        if (t1 >= n_verts || t2 >= n_verts || t3 >= n_verts)
            throw std::runtime_error("face indices out of bounds");


        vec3<float> v_0 = vec3<float>(verts_f[3*t1],verts_f[3*t1+1],verts_f[3*t1+2]);
        vec3<float> v_1 = vec3<float>(verts_f[3*t2],verts_f[3*t2+1],verts_f[3*t2+2]);
        vec3<float> v_2 = vec3<float>(verts_f[3*t3],verts_f[3*t3+1],verts_f[3*t3+2]);

        vec3<float> a(v_1-v_0);
        vec3<float> b(v_2-v_1);
        vec3<float> c(v_0-v_2);

        vec3<float> n = cross(a,b);

        const float eps = 1e-12*std::max(dot(a,a),std::max(dot(b,b),dot(c,c)));
        if (dot(n,n) < eps)
            {
            // the edges are colinear
            n = cross(a,vec3<float>(1,0,0));
            if (dot(n,n) < eps)
                {
                n = cross(a,vec3<float>(0,1,0));
                }
            if (dot(n,cross(b,c)) < 0) n = -n;
            }

        n = n / sqrtf(dot(n,n));

        // validate winding order
        if (dot(n,cross(b, c)) <= 0)
            {
            throw std::invalid_argument("triangles vertices must be counterclockwise and convex");
            }
        // store vertices
        m_vertices.push_back(v_0);
        m_vertices.push_back(v_1);
        m_vertices.push_back(v_2);

        m_face_normal.push_back(n);

        // find circumcenter and -radius
        //https://en.wikipedia.org/wiki/Circumscribed_circle
        float asq = dot(b,b); // a = BC
        float bsq = dot(c,c); // b = AC
        float csq = dot(a,a); // c = AB
        vec3<float> u = asq*(bsq+csq-asq)*v_0+bsq*(csq+asq-bsq)*v_1 + csq*(asq+bsq-csq)*v_2;
        u /= asq*(bsq+csq-asq)+bsq*(csq+asq-bsq)+csq*(asq+bsq-csq);
        m_face_origin[i] = u;

        float rsq = std::max(dot(v_0-u,v_0-u),std::max(dot(v_1-u,v_1-u),dot(v_2-u,v_2-u)));

        // precompute radius in the xy plane
        m_radius[i] = sqrtf(rsq);
        }

    // create the geometry
    m_geom_id = rtcNewUserGeometry(m_scene->getRTCScene(), N*n_faces);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1,0,1)));
    setOutlineMaterial(Material(RGB<float>(0,0,0), 1.0f));

    // register functions for embree
    rtcSetUserData(m_scene->getRTCScene(), m_geom_id, this);
    m_device->checkError();
    rtcSetBoundsFunction(m_scene->getRTCScene(), m_geom_id, &GeometryMesh::bounds);
    m_device->checkError();
    rtcSetIntersectFunction(m_scene->getRTCScene(), m_geom_id, &GeometryMesh::intersect);
    m_device->checkError();
    rtcSetOccludedFunction(m_scene->getRTCScene(), m_geom_id, &GeometryMesh::occlude);
    m_device->checkError();

    m_valid = true;
    }

GeometryMesh::~GeometryMesh()
    {
    }

/*! Compute the bounding box of a given primitive

    \param ptr Pointer to a GeometryMesh instance
    \param item Index of the primitive to compute the bounding box of
    \param bounds_o Output bounding box
*/
void GeometryMesh::bounds(void *ptr, size_t item, RTCBounds& bounds_o)
    {
    GeometryMesh *geom = (GeometryMesh*)ptr;
    unsigned int n_faces = geom->m_radius.size();
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    vec3<float> p3 = geom->m_position->get(i_poly);
    quat<float> q = geom->m_orientation->get(i_poly);
    vec3<float> o = geom->m_face_origin[i_face];

    // rotate into space frame
    vec3<float> r_t = p3 + rotate(q,o);
    float r = geom->m_radius[i_face];

    bounds_o.lower_x = r_t.x - r;
    bounds_o.lower_y = r_t.y - r;
    bounds_o.lower_z = r_t.z - r;

    bounds_o.upper_x = r_t.x + r;
    bounds_o.upper_y = r_t.y + r;
    bounds_o.upper_z = r_t.z + r;
    }

// From Real-time Collision Detection (Christer Ericson)
// Given ray pq and triangle abc, returns whether segment intersects
// triangle and if so, also returns the barycentric coordinates (u,v,w)
// of the intersection point
// Note: the triangle is assumed to be oriented counter-clockwise when viewed from the direction of p
inline bool IntersectRayTriangle(const vec3<float>& p, const vec3<float>& q,
     const vec3<float>& a, const vec3<float>& b, const vec3<float>& c,
    float &u, float &v, float &w, float &t)
    {
    vec3<float> ab = b - a;
    vec3<float> ac = c - a;
    vec3<float> qp = p - q;

    // Compute triangle normal. Can be precalculated or cached if
    // intersecting multiple segments against the same triangle
    vec3<float> n = cross(ab, ac);

    // Compute denominator d. If d <= 0, segment is parallel to or points
    // away from triangle, so exit early
    float d = dot(qp, n);
    if (d <= float(0.0)) return false;

    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    vec3<float> ap = p - a;
    t = dot(ap, n);
    if (t < float(0.0)) return false;
//    if (t > d) return false; // For segment; exclude this code line for a ray test

    // Compute barycentric coordinate components and test if within bounds
    vec3<float> e = cross(qp, ap);
    v = dot(ac, e);
    if (v < float(0.0) || v > d) return false;
    w = -dot(ab, e);
    if (w < float(0.0) || v + w > d) return false;

    // Segment/ray intersects triangle. Perform delayed division and
    // compute the last barycentric coordinate component
    float ood = float(1.0) / d;
    t *= ood;
    v *= ood;
    w *= ood;
    u = float(1.0) - v - w;
    return true;
    }

/*! Compute the intersection of a ray with the given primitive

    \param ptr Pointer to a GeometryMesh instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryMesh::intersect(void *ptr, RTCRay& ray, size_t item)
    {
    GeometryMesh *geom = (GeometryMesh*)ptr;

    unsigned int n_faces = geom->m_radius.size();
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    const vec3<float> p3 = geom->m_position->get(i_poly);
    const quat<float> q_world = geom->m_orientation->get(i_poly);
    const vec3<float> pos_world = p3 + rotate(q_world, geom->m_face_origin[i_face]);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), ray.dir);
    vec3<float> ray_org_local = rotate(conj(q_world), ray.org - pos_world);

    vec3<float> v0 = geom->m_vertices[i_face*3];
    vec3<float> v1 = geom->m_vertices[i_face*3+1];
    vec3<float> v2 = geom->m_vertices[i_face*3+2];
    float u,v,w,t;
    if (!IntersectRayTriangle(ray_org_local, ray_org_local+ray_dir_local, v0, v1, v2, u,v,w,t))
        return;
    // if the t is in (tnear,tfar), we hit the entry plane
    if ((ray.tnear < t) & (t < ray.tfar))
        {
        ray.tfar = t;
        ray.geomID = geom->m_geom_id;
        ray.primID = item;
        ray.Ng = rotate(q_world, geom->m_face_normal[i_face]);
        ray.shading_color = geom->m_color->get(i_poly);

        // determine distance from the hit point to the nearest edge
        vec3<float> edge;
        vec3<float> pt;
        if (u < v)
            {
            if (u < w)
                {
                edge = v2 - v1;
                pt = v1;
                }
            else
                {
                edge =  v1 - v0;
                pt = v0;
                }
            }
        else
            {
            if (v < w)
                {
                edge = v0 - v2;
                pt = v2;
                }
            else
                {
                edge = v1 - v0;
                pt = v0;
                }
            }

        // find the distance from the hit point to the line
        vec3<float> r_hit =  ray_org_local + t * ray_dir_local;
        vec3<float> q = cross(edge, r_hit - pt);
        float dsq = dot(q, q) / dot(edge,edge);
        ray.d = sqrtf(dsq);
        }
    }



/*! Test if a ray intersects with the given primitive

    \param ptr Pointer to a GeometryMesh instance
    \param ray The ray to intersect
    \param item Index of the primitive to compute the bounding box of
*/
void GeometryMesh::occlude(void *ptr, RTCRay& ray, size_t item)
    {
    // this method is a copy and pasted version of intersect with a different behavior on hit, to
    // meet Embree API standards. When intersect is updated, it should be copied and pasted back here.
    GeometryMesh *geom = (GeometryMesh*)ptr;

    unsigned int n_faces = geom->m_radius.size();
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    const vec3<float> p3 = geom->m_position->get(i_poly);
    const quat<float> q_world = geom->m_orientation->get(i_poly);
    const vec3<float> pos_world = p3 + rotate(q_world, geom->m_face_origin[i_face]);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), ray.dir);
    vec3<float> ray_org_local = rotate(conj(q_world), ray.org - pos_world);

    vec3<float> v0 = geom->m_vertices[i_face*3];
    vec3<float> v1 = geom->m_vertices[i_face*3+1];
    vec3<float> v2 = geom->m_vertices[i_face*3+2];
    float u,v,w,t;
    if (!IntersectRayTriangle(ray_org_local, ray_org_local+ray_dir_local, v0, v1, v2, u,v,w,t))
        return;

    ray.geomID = 0;
    }

/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, std::shared_ptr<GeometryMesh> >(m, "GeometryMesh", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             pybind11::array_t<unsigned int, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryMesh::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryMesh::getOrientationBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer)
        ;
    }

} } // end namespace fresnel::cpu
