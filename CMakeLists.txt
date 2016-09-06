cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(fresnel)

add_subdirectory(CMake)
include(CFlagsSetup)
include(PythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

#############################################################################################
# Options
option(ENABLE_TBB "Use TBB for threaded execution on the CPU" ON)
option(ENABLE_EMBREE "Use Embree for ray tracing on the CPU" ON)
option(ENABLE_CUDA "Use CUDA for GPU code execution" OFF)
option(ENABLE_OPTIX "Use OptiX for ray tracing on the GPU" OFF)

#############################################################################################
# Find libraries

if (ENABLE_EMBREE)
    find_package(Embree REQUIRED)
    include_directories(${Embree_INCLUDE_DIR})
endif()

if (ENABLE_TBB)
    find_package(TBB REQUIRED)
    include_directories(${TBB_INCLUDE_DIRS})
endif()

if (ENABLE_OPTIX)
    find_package(OptiX REQUIRED)
    include_directories(${OptiX_INCLUDE})
endif()

if (ENABLE_CUDA)
    find_package(CUDA 7.5 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
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
add_subdirectory(fresnel)
