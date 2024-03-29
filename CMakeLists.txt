cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fresnel LANGUAGES C CXX)

add_subdirectory(CMake)
include(fresnel-macros)

#################################
## CFLAGS configuration

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Set the default build type for the Ninja multi-configuration generator
if (CMAKE_GENERATOR MATCHES "Ninja Multi-Config")
    set(CMAKE_DEFAULT_BUILD_TYPE "Release")
endif()

# enable c++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Enable compiler warnings on gcc and clang (common compilers used by developers)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
endif()

# Enable color output from compiler
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
endif()

################################
## Python configuration

set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 2.2 CONFIG REQUIRED)
if (pybind11_FOUND)
    find_package_message(pybind11 "Found pybind11: ${pybind11_DIR} ${pybind11_INCLUDE_DIR} (version ${pybind11_VERSION})" "[${pybind11_DIR}][${pybind11_INCLUDE_DIR}]")
endif()

# when the user specifies CMAKE_INSTALL_PREFIX on first configure, install to "fresnel" under that prefix
set(PYTHON_SITE_INSTALL_DIR "fresnel" CACHE PATH
    "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)")

# when no CMAKE_INSTALL_PREFIX is set, default to python's install prefix
if((CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR "${PYTHON_PREFIX}" STREQUAL "${CMAKE_INSTALL_PREFIX}") AND PYTHON_SITE_PACKAGES)
    string(LENGTH "${PYTHON_PREFIX}" _python_prefix_len)
    string(SUBSTRING "${PYTHON_SITE_PACKAGES}" 0 ${_python_prefix_len} _python_site_package_prefix)
    math(EXPR _shart_char "${_python_prefix_len}+1")
    string(SUBSTRING "${PYTHON_SITE_PACKAGES}" ${_shart_char} -1 _python_site_package_rel)
    string(COMPARE EQUAL "${_python_site_package_prefix}" "${PYTHON_PREFIX}" _prefix_equal)
    if (NOT _prefix_equal)
        message(STATUS "Python site packages (${PYTHON_SITE_PACKAGES}) does not start with python prefix (${PYTHON_PREFIX})")
        message(STATUS "fresnel may not install to the correct location")
    endif()

    set(CMAKE_INSTALL_PREFIX "${PYTHON_PREFIX}" CACHE PATH "fresnel installation path" FORCE)
    set(PYTHON_SITE_INSTALL_DIR "${_python_site_package_rel}/fresnel" CACHE PATH
        "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)" FORCE)
endif()

find_package_message(fresnel_install "Installing fresnel python module to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}" "[${CMAKE_INSTALL_PREFIX}][${PYTHON_SITE_INSTALL_DIR}]")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

#############################################################################################
# Find libraries

option(ENABLE_EMBREE "Use Embree for ray tracing on the CPU" ON)

if (ENABLE_EMBREE)
    find_package_config_first(embree 4.0)
    if (embree_FOUND)
        find_package_message(embree "Found embree: ${embree_DIR} ${EMBREE_LIBRARY} ${EMBREE_INCLUDE_DIRS}" "[${EMBREE_LIBRARY}][${EMBREE_INCLUDE_DIRS}]")
    endif()

    find_package_config_first(TBB 4.3)

    if (TBB_FOUND)
        get_target_property(_tbb_library TBB::tbb IMPORTED_LOCATION_RELEASE)
        get_target_property(_tbb_include_dir TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
        find_package_message(tbb "Found TBB: ${TBB_DIR} ${_tbb_library} ${_tbb_include_dir}" "[${_tbb_library}][${_tbb_include_dir}]")

        # fix debug builds when the debug library is missing (such as in conda-forge installs)
        if (WIN32)
            set_target_properties(TBB::tbb PROPERTIES INTERFACE_COMPILE_DEFINITIONS "__TBB_NO_IMPLICIT_LINKAGE=1")
        endif()
    endif()

    if(EMBREE_LIBRARY AND NOT TARGET embree)
        add_library(embree UNKNOWN IMPORTED)
        set_target_properties(embree PROPERTIES
            IMPORTED_LOCATION "${EMBREE_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${EMBREE_INCLUDE_DIRS}"
            IMPORTED_LINK_INTERFACE_LIBRARIES "TBB::tbb")
    endif()

endif()

option(ENABLE_OPTIX "Use OptiX for ray tracing on the GPU" OFF)

if (ENABLE_OPTIX)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.0)
        message(SEND_ERROR "Fresnel requires CUDA 9.0 or newer")
    endif()

    find_package(OptiX REQUIRED)

    include_directories(${OptiX_INCLUDE})
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (NOT (ENABLE_OPTIX) AND NOT (ENABLE_EMBREE))
    message(SEND_ERROR "No libraries will be built. Enable at least one of embree or OptiX.")
endif()

#############################################################################################
### Add project subdirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/random123/examples)

add_subdirectory(fresnel)
