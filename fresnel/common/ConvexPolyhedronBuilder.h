// Copyright (c) 2016-2024 The Regents of the University of Michigan
// Part of fresnel, released under the BSD 3-Clause License.

#ifndef __CONVEX_POLYHEDRON_BUILDER_H__
#define __CONVEX_POLYHEDRON_BUILDER_H__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace fresnel
    {
//! Process a set of vertices
pybind11::dict find_polyhedron_faces(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> verts);

    } // namespace fresnel

#endif
