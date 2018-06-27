cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
project(fresnel)

add_subdirectory(CMake)
include(CFlagsSetup)
include(PythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

#############################################################################################
# Find libraries

option(ENABLE_EMBREE "Use Embree for ray tracing on the CPU" ON)

if (ENABLE_EMBREE)
    find_package(embree 3.0 CONFIG REQUIRED)
    if (embree_FOUND)
        find_package_message(embree "Found embree: ${embree_DIR} ${EMBREE_LIBRARY} ${EMBREE_INCLUDE_DIRS}" "[${EMBREE_LIBRARY}][${EMBREE_INCLUDE_DIRS}]")
    endif()

    find_package(TBB 4.3 REQUIRED)

    include_directories(${EMBREE_INCLUDE_DIRS})
    include_directories(${TBB_INCLUDE_DIR})
endif()

option(ENABLE_OPTIX "Use OptiX for ray tracing on the GPU" OFF)

if (ENABLE_OPTIX)
    find_package(OptiX REQUIRED)
    find_package(CUDA 7.5 REQUIRED)

    include_directories(${OptiX_INCLUDE})
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (NOT (ENABLE_OPTIX) AND NOT (ENABLE_EMBREE))
    message(SEND_ERROR "No libraries will be built. Enable at least one of embree or OptiX.")
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
