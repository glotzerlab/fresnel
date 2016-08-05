find_library(TBB_LIBRARY tbb
             PATHS ENV TBB_LINK)

get_filename_component(_tbb_lib_dir ${TBB_LIBRARY} DIRECTORY)

find_path(TBB_INCLUDE_DIR tbb/tbb.h
          PATHS ENV TBB_INC
          HINTS ${_tbb_lib_dir}/../include)

# handle the QUIETLY and REQUIRED arguments and set TBB_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_LIBRARY TBB_INCLUDE_DIR)

if(TBB_FOUND)
  set(TBB_LIBRARIES ${TBB_LIBRARY})
endif()
