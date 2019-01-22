// Copyright (c) 2016-2019 The Regents of the University of Michigan
// This file is part of the Fresnel project, released under the BSD 3-Clause License.

#include <memory>
#include <string>
#include <sstream>
#include <iomanip>

#include "Device.h"
#include "TracerDirect.h"
#include "TracerPath.h"
#include "TracerIDs.h"

using namespace std;

namespace fresnel { namespace gpu {

/*! Construct a new optix::Context with default options.

    \param ptx_root Directory where the PTX files are stored
    \param n Number of GPUs to use (-1 to use all).

    OptiX programs are loaded from PTX files, built from the .cu source files. These PTX files are stored in the
    python library directory. The Device instance tracks this directory for other classes (i.e. Tracer) to use
    when loading OptiX programs.
*/
Device::Device(const std::string& ptx_root, int n) : m_ptx_root(ptx_root)
    {
    // list the device ids to use
    int n_to_use = optix::ContextObj::getDeviceCount();
    if (n != -1)
        n_to_use = n;

    vector<int> devices;

    for (int i = 0; i < n_to_use; i++)
        devices.push_back(i);

    m_context = optix::Context::create();
    m_context->setDevices(devices.begin(), devices.end());

    m_context->setRayTypeCount(2);

    // miss programs
    optix::Program p2 = getProgram("_ptx_generated_path.cu.ptx", "path_miss");
    m_context->setMissProgram(TRACER_PATH_RAY_ID, p2);

    // initialize materials
    m_material = m_context->createMaterial();
    TracerDirect::setupMaterial(m_material, this);
    TracerPath::setupMaterial(m_material, this);
    }

/*! Destroy the underlying context
*/
Device::~Device()
    {
    // destroy programs
    for (auto elem : m_program_cache)
        {
        elem.second->destroy();
        }
    m_material->destroy();
    }

static std::string _formatOptiXDeviceList(const std::vector<int>& devices)
    {
    ostringstream s;

    for (const int& i : devices)
        {
        int sm = 0;
        optix::int2 cc;
        int clock_rate = 0;
        RTsize total_mem = 0;
        optix::ContextObj::getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(int), &sm);
        optix::ContextObj::getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(optix::int2), &cc);
        optix::ContextObj::getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(int), &clock_rate);
        optix::ContextObj::getDeviceAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(RTsize), &total_mem);

        float ghz = float(clock_rate)/1e6;
        int mib = int(float(total_mem)/(1024.0f*2014.0f));

        s << " [" + to_string(i) + "]: " <<
             setw(22) << optix::ContextObj::getDeviceName(i) << " " <<
             setw(4) << sm << " " <<
             "SM_" << cc.x << "." << cc.y << " ";
        s.precision(3);
        s.fill('0');

        s << "@ " << setw(4) << ghz << " GHz";
        s.fill(' ');
        s << ", " << setw(5) << mib << " MiB DRAM" << std::endl;
        }

    return s.str();
    }

/*! \returns Human readable string containing useful device information
*/
std::string Device::describe()
    {
    vector<int> devices = m_context->getEnabledDevices();
    return "Enabled OptiX devices:\n" + _formatOptiXDeviceList(devices);
    }

std::string Device::getAllGPUs()
    {
    vector<int> devices;

    for (unsigned int i = 0; i < optix::ContextObj::getDeviceCount(); i++)
        devices.push_back(i);

    return _formatOptiXDeviceList(devices);
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
        .def(pybind11::init<const std::string&, int>())
        .def("describe", &Device::describe)
        .def_static("getAllGPUs", &Device::getAllGPUs)
        ;
    }

} } // end namespace fresnel::gpu
