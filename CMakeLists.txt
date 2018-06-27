cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
project(fresnel)

add_subdirectory(CMake)
include(CFlagsSetup)
include(PythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

#############################################################################################
# Find libraries

find_package(embree 3.0 CONFIG)
if (embree_FOUND)
    find_package_message(embree "Found embree: ${embree_DIR} ${EMBREE_LIBRARY} ${EMBREE_INCLUDE_DIRS}" "[${EMBREE_LIBRARY}][${EMBREE_INCLUDE_DIRS}]")
endif()

option(ENABLE_EMBREE "Use Embree for ray tracing on the CPU" ${embree_FOUND})

if (ENABLE_EMBREE)
    include_directories(${EMBREE_INCLUDE_DIRS})
endif()

find_package(TBB 4.3)
option(ENABLE_TBB "Use TBB for threaded execution on the CPU" ${TBB_FOUND})

if (ENABLE_TBB)
    include_directories(${TBB_INCLUDE_DIR})
endif()

find_package(OptiX)
option(ENABLE_OPTIX "Use OptiX for ray tracing on the GPU" ${OptiX_FOUND})

if (ENABLE_OPTIX)
    include_directories(${OptiX_INCLUDE})
endif()

find_package(CUDA 7.5)
option(ENABLE_CUDA "Use CUDA for GPU code execution" ${CUDA_FOUND})

if (ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (NOT (ENABLE_CUDA AND ENABLE_OPTIX) AND NOT (ENABLE_EMBREE AND ENABLE_TBB))
    message(SEND_ERROR "No libraries will be built. Enable at least one of (embree and TBB) or (CUDA and OptiX).")
endif()

#############################################################################################
# Setup install location
set(PYTHON_MODULE_BASE_DIR fresnel)

#############################################################################################
### Add project subdirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/examples)
add_subdirectory(fresnel)
