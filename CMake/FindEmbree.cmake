find_library(Embree_LIBRARY embree
             PATHS ENV EMBREE_LINK)

get_filename_component(_embree_lib_dir ${Embree_LIBRARY} DIRECTORY)

find_path(Embree_INCLUDE_DIR embree2/rtcore.h
          PATHS ENV EMBREE_INC
          HINTS ${_embree_lib_dir}/../include)

# handle the QUIETLY and REQUIRED arguments and set Embree_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Embree DEFAULT_MSG Embree_LIBRARY Embree_INCLUDE_DIR)

if(Embree_FOUND)
  set(Embree_LIBRARIES ${Embree_LIBRARY})
endif()
