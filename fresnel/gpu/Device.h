// Copyright (c) 2016-2020 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include <map>
#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace fresnel
    {
namespace gpu
    {
//! Thin wrapper for optix::Context
/*! Handle construction and deletion of the optix context, and python lifetime as an exported class.

    Device also holds a cache of loaded OptiX programs so that other class instances can share them.
   On construction, the caller gives Device a directory in which to find ptx files. getProgram()
   gets a cached program (loading it if necessary), with the given file name and function
   definition.

    To facilitate multiple instances sharing a single entry point, Device also maintains a cache of
   entry points by program file and function name. getEntryPoint() will return the value in the
   cache, or create the entry point index and set the given program.

    optix::Material programs are an essential component to the ray tracing program flow. Due to the
   way OptiX is structured, materials must be assigned to geometry instances, however the material
   programs themselves are the domain of the ray tracer. Fresnel has a design where the Scene and
   its geometry are independent from the Tracer. To further complicate things, Geometry will need
   the material on initialization so that it can build the appropriate optix::GeometryInstance node.
   To solve these problems in a minimal way while introducing the fewest dependencies between
   classes, Device will hold on to the optix::Material objects for the various Tracer classes. Those
   objects will be initialized by static members of the Tracer so that as much code as possible
   related to the Tracer is in the Tracer class. When Geometry instances are created, the
   optix::Material is available from Device for assignment. To keep things simple, there will be
   only one material. The Tracer classes will use different ray id's to select the appropriate ray
   programs.

    These are the ID's assigned to the tracers (in TracerIDs.h):

      * TracerDirect: 0
      * TracerPath: 1

    Hopefully it is not too much of a performance degradation to have multiple ray id's loaded and
   only use one.
    TODO: test the performance impact of this.
*/
class Device
    {
    public:
    //! Constructor
    Device(const std::string& ptx_root, int n);

    //! Destructor
    ~Device();

    //! Access the context
    optix::Context& getContext()
        {
        return m_context;
        }

    //! Get information about this device
    std::string describe();

    //! List all GPUs
    static std::string getAllGPUs();

    //! Get a cached program
    optix::Program getProgram(const std::string& filename, const std::string& funcname);

    //! Get the entry point id of a given program
    unsigned int getEntryPoint(const std::string& filename, const std::string& funcname);

    //! Get the tracer material
    optix::Material getMaterial()
        {
        return m_material;
        }

    private:
    optix::Context m_context;   //!< Store the context
    std::string m_ptx_root;     //!< Directory where PTX files are stored
    optix::Material m_material; //!< Material for Direct ray tracer

    std::map<std::tuple<std::string, std::string>, optix::Program>
        m_program_cache; //!< The program cache
    std::map<std::tuple<std::string, std::string>, unsigned int>
        m_entrypoint_cache; //!< The entry point cache
    };

//! Export Device to python
void export_Device(pybind11::module& m);

    } // namespace gpu
    } // namespace fresnel

#endif
