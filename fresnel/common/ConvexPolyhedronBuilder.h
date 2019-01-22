// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef __CONVEX_POLYHEDRON_BUILDER_H__
#define __CONVEX_POLYHEDRON_BUILDER_H__

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace fresnel {

//! Process a set of vertices
pybind11::dict find_polyhedron_faces(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> verts);


}

#endif
