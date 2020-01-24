// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.
#include "ConvexPolyhedronBuilder.h"
#include "ColorMath.h"
#include "VectorMath.h"

#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/RboxPoints.h"

namespace fresnel {

/*! \param verts Vertices of the convex polyhedron

    Compute the convex hull of a set of vertices and return a python dictionary containing the face
   information needed for cpu::GeometryConvexPolyhedron and gpu::GeometryConvexPolyhedron.
*/
pybind11::dict find_polyhedron_faces(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> verts)
{
    // access the vertex data
    pybind11::buffer_info info_verts = verts.request();

    if (info_verts.ndim != 2)
        throw std::runtime_error("verts must be a 2-dimensional array");

    if (info_verts.shape[1] != 3)
        throw std::runtime_error("verts must be an N by 3 array");

    size_t N = info_verts.shape[0];
    double* coords = (double*)info_verts.ptr;

    // compute the convex hull
    orgQhull::Qhull q;
    q.runQhull("", 3, N, coords, "");

    // construct arrays of per-face data
    orgQhull::QhullFacetList facets = q.facetList();
    std::vector<vec3<float>> face_origin;
    std::vector<vec3<float>> face_normal;
    std::vector<RGB<float>> face_color;
    std::vector<unsigned int> face_sides;

    for (orgQhull::QhullFacet& f : facets)
    {
        orgQhull::QhullPoint o = f.getCenter();
        orgQhull::QhullHyperplane n = f.hyperplane();
        orgQhull::QhullVertexSet f_verts = f.vertices();

        face_origin.push_back(vec3<float>(o[0], o[1], o[2]));
        face_normal.push_back(vec3<float>(n[0], n[1], n[2]));
        face_color.push_back(RGB<float>(0.9f, 0.9f, 0.9f));
        face_sides.push_back(f_verts.count());
    }

    // determine radius
    vec3<double>* v = (vec3<double>*)info_verts.ptr;
    double radius = 0;
    for (unsigned int i = 0; i < N; i++)
    {
        double r = sqrt(dot(v[i], v[i]));
        radius = std::max(radius, r);
    }

    // pack return values in a python dict
    pybind11::dict retval;
    retval["face_origin"]
        = pybind11::array_t<float>({face_origin.size(), size_t(3)}, (float*)&face_origin[0]);
    retval["face_normal"]
        = pybind11::array_t<float>({face_normal.size(), size_t(3)}, (float*)&face_normal[0]);
    retval["face_color"]
        = pybind11::array_t<float>({face_color.size(), size_t(3)}, (float*)&face_color[0]);
    retval["face_sides"] = pybind11::array_t<unsigned int>(face_sides.size(), &face_sides[0]);
    retval["radius"] = radius;
    return retval;
}
} // namespace fresnel
