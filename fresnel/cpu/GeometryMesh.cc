// Copyright (c) 2016-2017 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "GeometryMesh.h"
#include "common/IntersectTriangle.h"


namespace fresnel { namespace cpu {

/*! \param scene Scene to attach the Geometry to
    \param vertices vertices of the mesh
    \param triangles triangle indices of the mesh
    \param N Number of polyhedra
    Initialize the mesh.
*/
GeometryMesh::GeometryMesh(std::shared_ptr<Scene> scene,
                             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> vertices,
                             unsigned int N)
    : Geometry(scene)
    {
    // allocate buffer data
    m_position = std::shared_ptr< Array< vec3<float> > >(new Array< vec3<float> >(N));
    m_orientation = std::shared_ptr< Array< quat<float>  > >(new Array< quat<float> >(N));

    // now create planes for each of the polygon edges
    pybind11::buffer_info info_vertices = vertices.request();

    if (info_vertices.ndim != 2)
        throw std::runtime_error("vertices must be a 2-dimensional array");

    if (info_vertices.shape[1] != 3)
        throw std::runtime_error("vertices must be a Nvert by 3 array");

    if (info_vertices.shape[0] % 3 != 0)
        throw std::runtime_error("the number of triangle vertices must be a multiple of three.");

    unsigned int n_faces = info_vertices.shape[0] / 3;
    unsigned int n_verts = info_vertices.shape[0];

    float *verts_f = (float *)info_vertices.ptr;

    m_radius.resize(n_faces,0.0);
    m_face_origin.resize(n_faces);

    for (unsigned int i = 0; i < n_faces; i++)
        {
        // construct the normal and origin of each plane
        unsigned int t1 = 3*i;
        unsigned int t2 = 3*i+1;
        unsigned int t3 = 3*i+2;

        if (t1 >= n_verts || t2 >= n_verts || t3 >= n_verts)
            throw std::runtime_error("face indices out of bounds");

        vec3<float> v_0 = vec3<float>(verts_f[3*t1],verts_f[3*t1+1],verts_f[3*t1+2]);
        vec3<float> v_1 = vec3<float>(verts_f[3*t2],verts_f[3*t2+1],verts_f[3*t2+2]);
        vec3<float> v_2 = vec3<float>(verts_f[3*t3],verts_f[3*t3+1],verts_f[3*t3+2]);

        vec3<float> a(v_1-v_0);
        vec3<float> b(v_2-v_1);
        vec3<float> c(v_0-v_2);

        vec3<float> n = cross(a,b);

        if (dot(n,cross(b, c)) == 0.0)
            {
//            throw std::invalid_argument("triangles vertices must not be colinear");
            }

        n = n / sqrtf(dot(n,n));

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

    m_color = std::shared_ptr< Array< RGB<float> > >(new Array< RGB<float> >(n_verts));

    // create the geometry
    m_geometry = rtcNewGeometry(m_device->getRTCDevice(), RTC_GEOMETRY_TYPE_USER);
    m_device->checkError();
    rtcSetGeometryUserPrimitiveCount(m_geometry,N*n_faces);
    m_device->checkError();
    m_geom_id = rtcAttachGeometry(m_scene->getRTCScene(), m_geometry);
    m_device->checkError();

    // set default material
    setMaterial(Material(RGB<float>(1,0,1)));
    setOutlineMaterial(Material(RGB<float>(0,0,0), 1.0f));

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

GeometryMesh::~GeometryMesh()
    {
    }

/*! Compute the bounding box of a given primitive

    \param args Arguments to the bounds check
*/
void GeometryMesh::bounds(const struct RTCBoundsFunctionArguments *args)
    {
    GeometryMesh *geom = (GeometryMesh*)args->geometryUserPtr;

    unsigned int item = args->primID;
    unsigned int n_faces = geom->m_radius.size();
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    vec3<float> p3 = geom->m_position->get(i_poly);
    quat<float> q = geom->m_orientation->get(i_poly);
    vec3<float> o = geom->m_face_origin[i_face];

    // rotate into space frame
    vec3<float> r_t = p3 + rotate(q,o);
    float r = geom->m_radius[i_face];

    RTCBounds& bounds_o = *args->bounds_o;
    bounds_o.lower_x = r_t.x - r;
    bounds_o.lower_y = r_t.y - r;
    bounds_o.lower_z = r_t.z - r;

    bounds_o.upper_x = r_t.x + r;
    bounds_o.upper_y = r_t.y + r;
    bounds_o.upper_z = r_t.z + r;
    }

/*! Compute the intersection of a ray with the given primitive
   \param args Arguments to the intersect check
*/
void GeometryMesh::intersect(const struct RTCIntersectFunctionNArguments *args)
    {
    GeometryMesh *geom = (GeometryMesh*)args->geometryUserPtr;

    unsigned int item = args->primID;
    unsigned int n_faces = geom->m_radius.size();
    unsigned int i_poly = item / n_faces;
    unsigned int i_face = item % n_faces;

    RTCRayHit& rayhit = *(RTCRayHit *)args->rayhit;
    RTCRay& ray = rayhit.ray;

    const vec3<float> p3 = geom->m_position->get(i_poly);
    const quat<float> q_world = geom->m_orientation->get(i_poly);

    // transform the ray into the primitive coordinate system
    vec3<float> ray_dir_local = rotate(conj(q_world), vec3<float>(ray.dir_x,ray.dir_y,ray.dir_z));
    vec3<float> ray_org_local = rotate(conj(q_world), vec3<float>(ray.org_x,ray.org_y,ray.org_z) - p3);

    vec3<float> v0 = geom->m_vertices[i_face*3];
    vec3<float> v1 = geom->m_vertices[i_face*3+1];
    vec3<float> v2 = geom->m_vertices[i_face*3+2];
    float u,v,w,t;
    if (!intersect_ray_triangle(ray_org_local, ray_org_local+ray_dir_local, v0, v1, v2, u,v,w,t))
        return;

    // if the t is in (tnear,tfar), we hit the entry plane
    if ((ray.tnear < t) & (t < ray.tfar))
        {
        rayhit.hit.u = 0.0f;
        rayhit.hit.v = 0.0f;
        ray.tfar = t;
        rayhit.hit.geomID = geom->m_geom_id;
        rayhit.hit.primID = item;
        vec3<float> n = rotate(q_world, geom->m_face_normal[i_face]);

        rayhit.hit.Ng_x = n.x;
        rayhit.hit.Ng_y = n.y;
        rayhit.hit.Ng_z = n.z;

        FresnelRTCIntersectContext & context = *(FresnelRTCIntersectContext *)args->context;
        rayhit.hit.instID[0] = context.context.instID[0];

        context.shading_color = geom->m_color->get(i_face*3 + 0)*u
            + geom->m_color->get(i_face*3 + 1)*v
            + geom->m_color->get(i_face*3 + 2)*w;

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
        context.d = sqrtf(dsq);
        }
    }


/*! \param m Python module to export in
 */
void export_GeometryMesh(pybind11::module& m)
    {
    pybind11::class_<GeometryMesh, std::shared_ptr<GeometryMesh> >(m, "GeometryMesh", pybind11::base<Geometry>())
        .def(pybind11::init<std::shared_ptr<Scene>,
             pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>,
             unsigned int>())
        .def("getPositionBuffer", &GeometryMesh::getPositionBuffer)
        .def("getOrientationBuffer", &GeometryMesh::getOrientationBuffer)
        .def("getColorBuffer", &GeometryMesh::getColorBuffer)
        ;
    }

} } // end namespace fresnel::cpu
