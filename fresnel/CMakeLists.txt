include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# variables needed for multi-config builds
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
file(RELATIVE_PATH relative_dir ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (_is_multi_config AND ${CMAKE_VERSION} VERSION_LESS "3.20.0")
    message(WARNING "Install fresnel to obtain a functioning Python package.\n"
                    "Use CMake 3.20 or newer to assemble a the Python package in ${PROJECT_BINARY_DIR}/<CONFIG>")
endif()

#####################################################
## C++ libraries

# _common
set(_common_sources
     common/module-common.cc
     common/ConvexPolyhedronBuilder.cc
    )

find_package(Qhull)

# work around broken libqhull installations
if (qhull_EMBED_SOURCE)
    list(APPEND _common_sources ${libqhullr_SOURCES} ${libqhullcpp_SOURCES})
endif()

pybind11_add_module(_common ${_common_sources})
if (_is_multi_config)
    set_target_properties(_common PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()
target_link_libraries(_common PRIVATE QHull::qhull_r QHull::qhull_cpp)
install(TARGETS _common LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

# _cpu
if (ENABLE_EMBREE)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_CPU)

if (WIN32)
target_compile_definitions(_common PUBLIC _USE_MATH_DEFINES R123_NO_SINCOS)
endif()

set(_cpu_sources
     cpu/module-cpu.cc
     cpu/Device.cc
     cpu/Scene.cc
     cpu/Geometry.cc
     cpu/GeometryCylinder.cc
     cpu/GeometryPolygon.cc
     cpu/GeometryMesh.cc
     cpu/GeometryConvexPolyhedron.cc
     cpu/GeometrySphere.cc
     cpu/Tracer.cc
     cpu/TracerDirect.cc
     cpu/TracerPath.cc
     cpu/Array.cc
   )

pybind11_add_module(_cpu "${_cpu_sources}")
target_link_libraries(_cpu PRIVATE embree TBB::tbb)
if (_is_multi_config)
    set_target_properties(_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()
if(APPLE)
    set_target_properties(_cpu PROPERTIES INSTALL_RPATH "@loader_path")
else()
    set_target_properties(_cpu PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if (WIN32)
target_compile_definitions(_cpu PUBLIC _USE_MATH_DEFINES R123_NO_SINCOS)
endif()

install(TARGETS _cpu LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

endif()

# _gpu
if (ENABLE_OPTIX)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_GPU)

# Build the PTX files for the OptiX programs
set(_ptx_files
    direct
    path
    GeometryCylinder
    GeometryMesh
    GeometrySphere
    GeometryPolygon
    GeometryConvexPolyhedron
    )

set(_ptx_sources "")
set(_ptx_outputs "")

foreach(ptx_file ${_ptx_files})
    list(APPEND _ptx_sources "gpu/${ptx_file}.cu")
    list(APPEND _ptx_outputs "${ptx_file}.ptx")
endforeach()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=code_is_unreachable")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

add_library(_ptx OBJECT ${_ptx_sources})
set_property(TARGET _ptx PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET _ptx PROPERTY CUDA_ARCHITECTURES 61-virtual)
target_compile_features(_ptx PUBLIC cxx_std_11)

add_custom_command(
  OUTPUT ${_ptx_outputs}
  COMMAND ${CMAKE_COMMAND}
    "-DOBJECTS=$<TARGET_OBJECTS:_ptx>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_wrapper.cmake
  VERBATIM
  DEPENDS _ptx ${_ptx_sources}
  COMMENT "Copying ptx files"
)

add_custom_target(copy_ptx ALL DEPENDS ${_ptx_outputs})

install(FILES $<TARGET_OBJECTS:_ptx>
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

# Build the shared object
set(_gpu_sources
     gpu/module-gpu.cc
     gpu/Device.cc
     gpu/Scene.cc
     gpu/Geometry.cc
     gpu/GeometryCylinder.cc
     gpu/GeometryMesh.cc
     gpu/GeometryPolygon.cc
     gpu/GeometryConvexPolyhedron.cc
     gpu/GeometrySphere.cc
     gpu/Tracer.cc
     gpu/TracerDirect.cc
     gpu/TracerPath.cc
     gpu/Array.cc
   )

pybind11_add_module(_gpu "${_gpu_sources}")
target_link_libraries(_gpu PRIVATE ${optix_LIBRARY})
if (_is_multi_config)
    set_target_properties(_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/$<CONFIG>/${relative_dir})
endif()
if(APPLE)
    set_target_properties(_gpu PROPERTIES INSTALL_RPATH "@loader_path")
else()
    set_target_properties(_gpu PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

if (WIN32)
target_compile_definitions(_gpu PUBLIC _USE_MATH_DEFINES)
endif()

install(TARGETS _gpu LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})
endif()

#####################################################
### Python only modules

set(files __init__.py
          conftest.py
          geometry.py
          tracer.py
          material.py
          camera.py
          util.py
          color.py
          light.py
          interact.py
          version.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

copy_files_to_build("${files}" "fresnel" "*.py")

add_subdirectory(test)
