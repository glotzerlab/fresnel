find_library(Qhull_r_LIBRARY qhull_r)

get_filename_component(_qhull_lib_dir "${Qhull_r_LIBRARY}" DIRECTORY)

find_path(Qhull_INCLUDE_DIR libqhull_r/libqhull_r.h
          HINTS ${_qhull_lib_dir}/../include)

find_library(Qhull_cpp_LIBRARY qhullcpp
             HINTS ${_qhull_lib_dir})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qhull
                                  REQUIRED_VARS Qhull_r_LIBRARY Qhull_INCLUDE_DIR)


if(Qhull_r_LIBRARY AND Qhull_cpp_LIBRARY)
    set(CMAKE_REQUIRED_LIBRARIES "${Qhull_cpp_LIBRARY}; ${Qhull_r_LIBRARY}")
    set(CMAKE_REQUIRED_INCLUDES "${Qhull_INCLUDE_DIR}")
    check_cxx_source_compiles("#include \"libqhullcpp/Qhull.h\"\nint main(int argc, char **argv) { orgQhull::Qhull q; return 0;}" can_link_libqhullcpp)

    if(can_link_libqhullcpp AND NOT TARGET QHull::qhull_cpp)
        add_library(QHull::qhull_cpp UNKNOWN IMPORTED)
        set_target_properties(QHull::qhull_cpp PROPERTIES
            IMPORTED_LOCATION "${Qhull_cpp_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Qhull_INCLUDE_DIR}")
    endif()

    if(can_link_libqhullcpp AND NOT TARGET QHull::qhull_r)
        add_library(QHull::qhull_r UNKNOWN IMPORTED)
        set_target_properties(QHull::qhull_r PROPERTIES
            IMPORTED_LOCATION "${Qhull_r_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Qhull_INCLUDE_DIR}")
    endif()
else()
    set(can_link_libqhullcpp "FALSE")
endif()

# build libqhullcpp if needed
if(NOT can_link_libqhullcpp)
    set(
        libqhullr_SOURCES
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/global_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/stat_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/geom2_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/poly2_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/merge_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/libqhull_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/geom_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/poly_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/qset_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/mem_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/random_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/usermem_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/io_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/user_r.c
            ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhull_r/rboxlib_r.c
    )

    set(libqhullcpp_SOURCES
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/Coordinates.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/PointCoordinates.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/Qhull.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacet.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacetList.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullFacetSet.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullHyperplane.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPoint.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPointSet.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullPoints.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullQh.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullRidge.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullSet.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullStat.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullVertex.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/QhullVertexSet.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RboxPoints.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RoadError.cpp
        ${CMAKE_SOURCE_DIR}/extern/qhull/src/libqhullcpp/RoadLogEvent.cpp
        )

    add_library(qhull_cpp STATIC ${libqhullcpp_SOURCES})
    set_target_properties(qhull_cpp PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/extern/qhull/src)
    set_target_properties(qhull_cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/extern/qhull/src)
    set_target_properties(qhull_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    add_library(QHull::qhull_cpp ALIAS qhull_cpp)

    add_library(qhull_r STATIC ${libqhullr_SOURCES})
    set_target_properties(qhull_r PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/extern/qhull/src)
    set_target_properties(qhull_r PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/extern/qhull/src)
    set_target_properties(qhull_r PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    add_library(QHull::qhull_r ALIAS qhull_r)
endif()
