# use pybind11's tools to find python and create python modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/pybind11/tools)

# trick pybind11 tools to allow us to manage C++ flags
# cmake ignores this in 2.8, but when pybind11 sees this
# it will not override fresnel's cflags
set(CMAKE_CXX_STANDARD 11)

# frenel's clfags setup script will take care of proper cxx flags settings
include(pybind11Tools)
