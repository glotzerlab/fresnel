find_library(Qhull_r_LIBRARY qhull_r)

get_filename_component(_qhull_lib_dir "${Qhull_r_LIBRARY}" DIRECTORY)

find_path(Qhull_INCLUDE_DIR libqhull_r/libqhull_r.h
          HINTS ${_qhull_lib_dir}/../include)

find_library(Qhull_cpp_LIBRARY qhullcpp
             HINTS ${_qhull_lib_dir})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qhull
                                  REQUIRED_VARS Qhull_r_LIBRARY Qhull_cpp_LIBRARY Qhull_INCLUDE_DIR)


if(Qhull_r_LIBRARY AND NOT TARGET QHull::qhull_r)
    add_library(QHull::qhull_r UNKNOWN IMPORTED)
    set_target_properties(QHull::qhull_r PROPERTIES
        IMPORTED_LOCATION "${Qhull_r_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Qhull_INCLUDE_DIR}")
endif()

if(Qhull_cpp_LIBRARY AND NOT TARGET QHull::qhull_cpp)
    add_library(QHull::qhull_cpp UNKNOWN IMPORTED)
    set_target_properties(QHull::qhull_cpp PROPERTIES
        IMPORTED_LOCATION "${Qhull_cpp_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Qhull_INCLUDE_DIR}")
endif()
