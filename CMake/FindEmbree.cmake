find_library(Embree_LIBRARY embree
             HINTS ENV EMBREE_LINK)

get_filename_component(_embree_lib_dir ${Embree_LIBRARY} DIRECTORY)

find_path(Embree_INCLUDE_DIR embree2/rtcore.h
          HINTS ${_embree_lib_dir}/../include)

if(Embree_INCLUDE_DIR AND EXISTS "${Embree_INCLUDE_DIR}/embree2/rtcore.h")
    file(STRINGS "${Embree_INCLUDE_DIR}/embree2/rtcore.h" Embree_H REGEX "^#define RTCORE_VERSION_.*$")

    string(REGEX REPLACE ".*#define RTCORE_VERSION_MAJOR ([0-9]+).*$" "\\1" Embree_VERSION_MAJOR "${Embree_H}")
    string(REGEX REPLACE "^.*RTCORE_VERSION_MINOR ([0-9]+).*$" "\\1" Embree_VERSION_MINOR  "${Embree_H}")
    string(REGEX REPLACE "^.*RTCORE_VERSION_PATCH ([0-9]+).*$" "\\1" Embree_VERSION_PATCH "${Embree_H}")
    set(Embree_VERSION_STRING "${Embree_VERSION_MAJOR}.${Embree_VERSION_MINOR}.${Embree_VERSION_PATCH}")
endif()

if(Embree_INCLUDE_DIR AND EXISTS "${Embree_INCLUDE_DIR}/embree2/rtcore_version.h")
    file(STRINGS "${Embree_INCLUDE_DIR}/embree2/rtcore_version.h" Embree_H REGEX "^#define RTCORE_VERSION_.*$")

    string(REGEX REPLACE ".*#define RTCORE_VERSION_MAJOR ([0-9]+).*$" "\\1" Embree_VERSION_MAJOR "${Embree_H}")
    string(REGEX REPLACE "^.*RTCORE_VERSION_MINOR ([0-9]+).*$" "\\1" Embree_VERSION_MINOR  "${Embree_H}")
    string(REGEX REPLACE "^.*RTCORE_VERSION_PATCH ([0-9]+).*$" "\\1" Embree_VERSION_PATCH "${Embree_H}")
    set(Embree_VERSION_STRING "${Embree_VERSION_MAJOR}.${Embree_VERSION_MINOR}.${Embree_VERSION_PATCH}")
endif()

# handle the QUIETLY and REQUIRED arguments and set Embree_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Embree
                                  REQUIRED_VARS Embree_LIBRARY Embree_INCLUDE_DIR
                                  VERSION_VAR Embree_VERSION_STRING)
