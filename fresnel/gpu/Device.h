// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#ifndef DEVICE_H_
#define DEVICE_H_

#include <optixu/optixpp_namespace.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <map>

// setup pybind11 to use std::shared_ptr
PYBIND11_DECLARE_HOLDER_TYPE(T_shared_ptr_bind, std::shared_ptr<T_shared_ptr_bind>);

namespace fresnel { namespace gpu {

//! Thin wrapper for optix::Context
/*! Handle construction and deletion of the optix context, and python lifetime as an exported class.

    Device also holds a cache of loaded OptiX programs so that other class instances can share them. On construction,
    the caller gives Device a directory in which to find ptx files. getProgram() gets a cached program (loading
    it if necessary), with the given file name and function definition.

    To facilitate multiple instances sharing a single entry point, Device also maintains a cache of entry points
    by program file and function name. getEntryPoint() will return the value in the cache, or create the entry point
    index and set the given program.
*/
class Device
    {
    public:
        //! Constructor
        Device(const std::string& ptx_root);

        //! Destructor
        ~Device();

        //! Access the context
        optix::Context& getContext()
            {
            return m_context;
            }

        //! Get information about this device
        std::string getStats();

        //! Get a cached program
        optix::Program getProgram(const std::string& filename, const std::string& funcname);

        //! Get the entry point id of a given program
        unsigned int getEntryPoint(const std::string& filename, const std::string& funcname);

    private:
        optix::Context m_context; //!< Store the context
        std::string m_ptx_root;   //!< Directory where PTX files are stored
        std::map< std::tuple<std::string, std::string>, optix::Program> m_program_cache;    //!< The program cache
        std::map< std::tuple<std::string, std::string>, unsigned int> m_entrypoint_cache;   //!< The entry point cache
    };

//! Export Device to python
void export_Device(pybind11::module& m);

} } // end namespace fresnel::gpu

#endif
