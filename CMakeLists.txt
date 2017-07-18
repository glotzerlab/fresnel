cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(fresnel)

add_subdirectory(CMake)
include(CFlagsSetup)
include(PythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

#############################################################################################
# Find libraries

find_package(Embree 2.10.0)
option(ENABLE_EMBREE "Use Embree for ray tracing on the CPU" ${Embree_FOUND})

if (ENABLE_EMBREE)
    include_directories(${Embree_INCLUDE_DIR})
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

if (INSTALL_SITE)
    SET(CMAKE_INSTALL_PREFIX ${PYTHON_SYSTEM_SITE} CACHE PATH "Python site installation directory" FORCE)
    message(STATUS "Setting installation site dir: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

set(PYTHON_MODULE_BASE_DIR fresnel)

#############################################################################################
### Add project subdirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/examples)
add_subdirectory(fresnel)
