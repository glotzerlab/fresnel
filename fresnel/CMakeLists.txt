include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#####################################################
## C++ libraries

# _common
set(_common_sources
     common/module-common.cc
     common/ConvexPolyhedronBuilder.cc
    )

find_package(Qhull)

pybind11_add_module(_common ${_common_sources})
target_link_libraries(_common PRIVATE QHull::qhull_r QHull::qhull_cpp)
install(TARGETS _common LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})

# _cpu
if (ENABLE_EMBREE)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_CPU)

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
target_link_libraries(_cpu PRIVATE EMBREE::embree)
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch compute_30 --use_fast_math")

add_library(_ptx OBJECT ${_ptx_sources})
set_property(TARGET _ptx PROPERTY CUDA_PTX_COMPILATION ON)
target_compile_features(_ptx PUBLIC cxx_std_11)

add_custom_command(
  OUTPUT ${_ptx_outputs}
  COMMAND ${CMAKE_COMMAND}
    "-DOBJECTS=$<TARGET_OBJECTS:_ptx>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_wrapper.cmake
  VERBATIM
  DEPENDS $<TARGET_OBJECTS:_ptx>
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
install(TARGETS _gpu LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR})
endif()

#####################################################
### Python only modules

# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy ${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          geometry.py
          tracer.py
          material.py
          camera.py
          util.py
          color.py
          light.py
          interact.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_python ALL DEPENDS ${files})
