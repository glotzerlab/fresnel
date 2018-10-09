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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src)

#############################################################################################
### Qhull compilation

set(
    libqhullr_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/global_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/stat_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/geom2_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/poly2_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/merge_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/libqhull_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/geom_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/poly_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/qset_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/mem_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/random_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/usermem_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/io_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/user_r.c
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhull_r/rboxlib_r.c
)

set(
    libqhullcpp_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/Coordinates.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/PointCoordinates.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/Qhull.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacetList.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacetSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullHyperplane.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPointSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPoints.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullQh.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullRidge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullStat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullVertex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullVertexSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RboxPoints.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RoadError.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RoadLogEvent.cpp
)

add_subdirectory(fresnel)
