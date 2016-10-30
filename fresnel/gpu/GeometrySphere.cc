// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <stdexcept>
#include <pybind11/stl.h>
#include <optixu/optixu_math_namespace.h>

#include "GeometrySphere.h"

namespace fresnel { namespace gpu {

	/*! \param scene Scene to attach the Geometry to
	  \param position position of each sphere
	  \param radius radius of each sphere
	  Initialize the sphere.
	  */

	GeometrySphere::GeometrySphere(std::shared_ptr<Scene> scene,
			const std::vector<std::tuple<float, float, float> > &position,
			const std::vector< float > &radius)
		: Geometry(scene)
		{
		std::cout << "Create GeometrySphere" << std::endl;

		// Declared initial variabls
		optix::Program intersection_program;
		optix::Program bounding_box_program;

		// copy data into the local buffers
		if (radius.size() != position.size())
			throw std::invalid_argument("radius must have the same length as position");
		m_position.resize(position.size());
		m_radius.resize(position.size());

		for (unsigned int i = 0; i < position.size(); i++)
			{
			m_position[i] = vec3<float>(std::get<0>(position[i]), std::get<1>(position[i]), std::get<2>(position[i]));
			m_radius[i] = float (radius[i]);
			}

		auto device = scene->getDevice();
		auto context = device->getContext();
		m_geometry = context->createGeometry();
		m_geometry->setPrimitiveCount(m_position.size());

		const char * path_to_ptx = "_ptx_generated_GeometrySphere.cu.ptx";
		bounding_box_program = device->getProgram(path_to_ptx, "bounds");
		m_geometry->setBoundingBoxProgram(bounding_box_program);

		intersection_program = device->getProgram(path_to_ptx, "intersect");
		m_geometry->setIntersectionProgram(intersection_program);

		optix::Buffer optix_positions = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_position.size());
		m_geometry["spheres"]->setBuffer(optix_positions);
		optix::float4 *position_buffer = (optix::float4 *) optix_positions->map();
		for(unsigned int i = 0; i < m_position.size(); i++)
			{
			position_buffer[i] = optix::make_float4(m_position[i].x, m_position[i].y, m_position[i].z, m_radius[i]); 
			}
		optix_positions->unmap();
		setupInstance();
		}

	GeometrySphere::~GeometrySphere()
		{
		std::cout << "Destroy GeometrySphere" << std::endl;
		}

	void export_GeometrySphere(pybind11::module& m)
		{
		pybind11::class_<GeometrySphere, std::shared_ptr<GeometrySphere>>(m, "GeometrySphere", pybind11::base<Geometry>())
			.def(pybind11::init<std::shared_ptr<Scene>,
				 const std::vector<std::tuple<float, float, float>> &,
				 const std::vector<float> &
				 >())
			;
		}

} } // end namespace fresnel::gpu
