include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#####################################################
## C++ libraries

# _common
set(_common_sources
     common/module-common.cc
     common/ConvexPolyhedronBuilder.cc
    )

pybind11_add_module(_common ${_common_sources} ${libqhullr_SOURCES} ${libqhullcpp_SOURCES})
set_source_files_properties(${libqhullr_SOURCES} ${libqhullcpp_SOURCES} PROPERTIES COMPILE_FLAGS -w) # hide compiler warnings for qhull
install(TARGETS _common LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR})

# _cpu
if (ENABLE_EMBREE)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_CPU)

set(_cpu_sources
     cpu/module-cpu.cc
     cpu/Device.cc
     cpu/Scene.cc
     cpu/Geometry.cc
     cpu/GeometryCylinder.cc
     cpu/GeometryPrism.cc
     cpu/GeometryMesh.cc
     cpu/GeometryConvexPolyhedron.cc
     cpu/GeometrySphere.cc
     cpu/Tracer.cc
     cpu/TracerDirect.cc
     cpu/TracerPath.cc
     cpu/Array.cc
   )

pybind11_add_module(_cpu "${_cpu_sources}")
target_link_libraries(_cpu PRIVATE ${EMBREE_LIBRARY} ${TBB_LIBRARIES})
install(TARGETS _cpu LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR})

endif()

# _gpu
if (ENABLE_OPTIX)

target_compile_definitions(_common PUBLIC FRESNEL_BUILD_GPU)

# Build the PTX files for the OptiX programs
set(_ptx_sources
	gpu/direct.cu
    gpu/path.cu
    gpu/GeometryCylinder.cu
	gpu/GeometrySphere.cu
    gpu/GeometryPrism.cu
    gpu/GeometryConvexPolyhedron.cu)

list(APPEND CUDA_NVCC_FLAGS "-Xcudafe;--diag_suppress=code_is_unreachable")
list(APPEND CUDA_NVCC_FLAGS "-arch;compute_30;--use_fast_math")
CUDA_WRAP_SRCS(_ptx PTX generated_ptx_files ${_ptx_sources})
add_custom_target(_ptx ALL
    DEPENDS ${generated_ptx_files} ${_ptx_sources}
    SOURCES ${_ptx_sources})

install(FILES ${generated_ptx_files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

# Build the shared object
set(_gpu_sources
     gpu/module-gpu.cc
     gpu/Device.cc
     gpu/Scene.cc
     gpu/Geometry.cc
     gpu/GeometryCylinder.cc
     gpu/GeometryPrism.cc
     gpu/GeometryConvexPolyhedron.cc
     gpu/GeometrySphere.cc
     gpu/Tracer.cc
     gpu/TracerDirect.cc
     gpu/TracerPath.cc
     gpu/Array.cc
   )

pybind11_add_module(_gpu "${_gpu_sources}")
target_link_libraries(_gpu PRIVATE ${optix_LIBRARY})
install(TARGETS _gpu LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR})
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
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_python ALL DEPENDS ${files})
