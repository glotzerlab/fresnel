find_path(EMBREE_INCLUDE_DIR embree4/rtcore.h)

find_library(EMBREE_LIBRARY embree4
             HINTS ${EMBREE_INCLUDE_DIR}/../lib )

if(EMBREE_INCLUDE_DIR AND EXISTS "${EMBREE_INCLUDE_DIR}/embree4/rtcore_version.h")
    file(STRINGS "${EMBREE_INCLUDE_DIR}/embree4/rtcore_version.h" EMBREE_H REGEX "^#define EMBREE_VERSION_.*$")
    string(REGEX REPLACE ".*#define EMBREE_VERSION_MAJOR ([0-9]+).*$" "\\1" EMBREE_VERSION_MAJOR "${EMBREE_H}")
    string(REGEX REPLACE "^.*EMBREE_VERSION_MINOR ([0-9]+).*$" "\\1" EMBREE_VERSION_MINOR  "${EMBREE_H}")
    string(REGEX REPLACE "^.*EMBREE_VERSION_PATCH ([0-9]+).*$" "\\1" EMBREE_VERSION_PATCH  "${EMBREE_H}")
    set(EMBREE_VERSION_STRING "${EMBREE_VERSION_MAJOR}.${EMBREE_VERSION_MINOR}.${EMBREE_VERSION_PATH}")
endif()

# handle the QUIETLY and REQUIRED arguments and set EMBREE_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(embree
                                  REQUIRED_VARS EMBREE_LIBRARY EMBREE_INCLUDE_DIR
                                  VERSION_VAR EMBREE_VERSION_STRING)

if(EMBREE_LIBRARY AND NOT TARGET embree)
    add_library(embree UNKNOWN IMPORTED)
    set_target_properties(embree PROPERTIES
        IMPORTED_LOCATION "${EMBREE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${EMBREE_INCLUDE_DIR}")
endif()

set(EMBREE_INCLUDE_DIRS ${EMBREE_INCLUDE_DIR})
