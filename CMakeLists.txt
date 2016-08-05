cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)
project(fresnel)

add_subdirectory(CMake)
include(CFlagsSetup)
include(PythonSetup)
include_directories(${PYTHON_INCLUDE_DIR})

#############################################################################################
# Options
option(ENABLE_TRACE_CPU "Enable raytracing on the CPU" ON)

#############################################################################################
# Find libraries

if (ENABLE_TRACE_CPU)
    find_package(Embree REQUIRED)
    include_directories(${Embree_INCLUDE_DIR})

    find_package(TBB REQUIRED)
    include_directories(${TBB_INCLUDE_DIRS})
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
