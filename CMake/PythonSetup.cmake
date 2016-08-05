#########################################################################################
### Find Python and set PYTHON_SITEDIR, the location to install python modules
# macro for running python and getting output
macro(run_python code result)
execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} -c ${code}
    OUTPUT_VARIABLE ${result}
    RESULT_VARIABLE PY_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if(PY_ERR)
    message(STATUS "Error while querying python for information")
endif(PY_ERR)
endmacro(run_python)

# find the python interpreter
find_program(PYTHON_EXECUTABLE NAMES python3 python)

# get the python installation prefix and version
run_python("import sys\; print('%d' % (sys.version_info[0]))" PYTHON_VERSION_MAJOR)
run_python("import sys\; print('%d' % (sys.version_info[1]))" PYTHON_VERSION_MINOR)
set(PYTHON_VERSION "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
string(REPLACE "." "" _python_version_no_dots ${PYTHON_VERSION})

# determine the include directory
if (PYTHON_VERSION VERSION_GREATER 3)
    run_python("import sysconfig\; print(sysconfig.get_path('include'))" _python_include_hint)
    run_python("import sysconfig\; print(sysconfig.get_config_var('LIBDIR'))" _python_lib_hint)
    run_python("import sysconfig\; print(sysconfig.get_config_var('LIBPL'))" _python_static_hint)

    if (ENABLE_STATIC)
        run_python("import sysconfig\; print(sysconfig.get_config_var('LIBRARY'))" _python_dynamic_lib_name)
    else()
        run_python("import sysconfig\; print(sysconfig.get_config_var('LDLIBRARY'))" _python_dynamic_lib_name)
    endif()
else()
    run_python("from distutils import sysconfig\; print sysconfig.get_python_inc()" _python_include_hint)
    run_python("from distutils import sysconfig\; print sysconfig.PREFIX" _python_prefix_hint)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBPL')" _python_static_hint)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBRARY')" _python_static_lib_name)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LDLIBRARY')" _python_dynamic_lib_name)
endif()

find_path(PYTHON_INCLUDE_DIR Python.h
          HINTS ${_python_include_hint}
          NO_DEFAULT_PATH)

# determine the packages directory
run_python("from distutils import sysconfig\; print(sysconfig.get_python_lib( plat_specific=True))" PYTHON_SYSTEM_SITE)
run_python("import site\; print(site.USER_SITE)" PYTHON_USER_SITE)

# find the python library
# add a blank suffix to the beginning to find the Python framework
set(_old_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ";${CMAKE_FIND_LIBRARY_SUFFIXES}")
find_library(PYTHON_LIBRARY
             NAMES python${_python_version_no_dots} python${PYTHON_VERSION} python${PYTHON_VERSION}m
             HINTS ${_python_prefix_hint} ${_python_lib_hint}
             PATH_SUFFIXES lib64 lib libs
             NO_DEFAULT_PATH
             )
set(${CMAKE_FIND_LIBRARY_SUFFIXES} _old_suffixes)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Python DEFAULT_MSG PYTHON_EXECUTABLE PYTHON_LIBRARY PYTHON_INCLUDE_DIR PYTHON_SYSTEM_SITE PYTHON_USER_SITE)

#### Setup numpy
# if (PYTHON_VERSION VERSION_GREATER 3)
#     run_python("import numpy\; print(numpy.get_include())" NUMPY_INCLUDE_GUESS)
# else()
#     run_python("import numpy\; print numpy.get_include()" NUMPY_INCLUDE_GUESS)
# endif()

# # We use the full path name (including numpy on the end), but
# # Double-check that all is well with that choice.
# find_path(
#     NUMPY_INCLUDE_DIR
#     numpy/arrayobject.h
#     HINTS
#     ${NUMPY_INCLUDE_GUESS}
#     )

# FIND_PACKAGE_HANDLE_STANDARD_ARGS(numpy DEFAULT_MSG NUMPY_INCLUDE_DIR)

# if (NUMPY_INCLUDE_DIR)
# mark_as_advanced(NUMPY_INCLUDE_DIR)
# endif (NUMPY_INCLUDE_DIR)

# include_directories(${NUMPY_INCLUDE_DIR})
#add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)
# add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

#############################################################################################
# Fixup conda linking, if this python appears to be a conda python
get_filename_component(_python_bin_dir ${PYTHON_EXECUTABLE} DIRECTORY)
if (EXISTS "${_python_bin_dir}/conda")
    message("-- Detected conda python, activating workaround")
    set(_using_conda On)
else()
    set(_using_conda Off)
endif()

macro(fix_conda_python target)
if (_using_conda)
get_filename_component(_python_lib_file ${PYTHON_LIBRARY} NAME)
add_custom_command(TARGET ${target} POST_BUILD
                          COMMAND install_name_tool ARGS -change ${_python_lib_file} ${PYTHON_LIBRARY} $<TARGET_FILE:${target}>)
endif ()
endmacro()
