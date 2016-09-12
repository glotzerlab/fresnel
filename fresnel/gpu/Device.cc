// Copyright (c) 2016 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <memory>
#include <string>

#include "Device.h"
#include "TracerWhitted.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! Construct a new optix::Context with default options.

    \param ptx_root Directory where the PTX files are stored

    OptiX programs are loaded from PTX files, built from the .cu source files. These PTX files are stored in the
    python library directory. The Device instance tracks this directory for other classes (i.e. Tracer) to use
    when loading OptiX programs.
*/
Device::Device(const std::string& ptx_root) : m_ptx_root(ptx_root)
    {
    std::cout << "Create GPU Device" << std::endl;
    m_context = optix::Context::create();

    // initialize materials
    m_whitted_mat = m_context->createMaterial();
    TracerWhitted::setupMaterial(m_whitted_mat, this);
    }

/*! Destroy the underlying context
*/
Device::~Device()
    {
    std::cout << "Destroy GPU Device" << std::endl;
    m_whitted_mat->destroy();
    m_context->destroy();
    }

/*! \returns Human readable string containing useful device information
*/
std::string Device::getStats()
    {
    std::string s("OptiX devices:");

    vector<int> devices = m_context->getEnabledDevices();

    for (const int& i : devices)
        {
        s += "\n[" + to_string(i) + "]: " + m_context->getDeviceName(i);
        }

    return s;
    }


/*! \param filename Name of the file to load the program from
    \param funcname Name of the function to load
    \returns An optix::Program loaded from the given function and file name.

    A file `filename` is assumed to exist in the ptx root dir. If this program has already been loaded,
    a shared pointer to it is returned. If it has not yet been loaded, it will be loaded first.
*/
optix::Program Device::getProgram(const std::string& filename, const std::string& funcname)
    {
    // return the cached program if it has already been loaded
    auto search = m_program_cache.find(make_tuple(filename, funcname));
    if (search != m_program_cache.end())
        {
        return search->second;
        }
    else
        {
        // if it is not found, load the program and return it.
        optix::Program p = m_context->createProgramFromPTXFile(m_ptx_root + "/" + filename, funcname);
        m_program_cache[make_tuple(filename, funcname)] = p;
        return p;
        }
    }

/*! \param filename Name of the file to load the program from
    \param funcname Name of the function to load
    \returns The entry point index for the program

    If the entry point for this program already exists in the cache, return it. If not, load the program from the
    cache, create an entry point and set the entry point program first.
*/
unsigned int Device::getEntryPoint(const std::string& filename, const std::string& funcname)
    {
    // search for the given program in the cache
    auto search = m_entrypoint_cache.find(make_tuple(filename, funcname));
    if (search != m_entrypoint_cache.end())
        {
        // if found, return the cached entry point index
        return search->second;
        }
    else
        {
        // if it is not found, load the program from the cache and set it to the next entry point index
        optix::Program p = getProgram(filename, funcname);
        unsigned int entry = m_context->getEntryPointCount();
        m_context->setEntryPointCount(entry + 1);
        m_context->setRayGenerationProgram(entry, p);
        return entry;
        }
    }

/*! \param m Python module to export in
 */
void export_Device(pybind11::module& m)
    {
    pybind11::class_<Device, std::shared_ptr<Device> >(m, "Device")
        .def(pybind11::init<const std::string&>())
        .def("getStats", &Device::getStats)
        ;
    }

} } // end namespace fresnel::gpu
