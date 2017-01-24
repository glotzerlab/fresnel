include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#####################################################
### Macro for building python module

macro(add_python_module target sources)
# build the shared library
add_library (${target} SHARED ${sources})
# change the name of the library to be a valid python module
set_target_properties(${target} PROPERTIES PREFIX "" OUTPUT_NAME "${target}")

# fixup the library to be a working python module on macOS
if(APPLE)
    set_target_properties(${target} PROPERTIES SUFFIX ".so")
    target_link_libraries(${target} ${PYTHON_LIBRARY})
    fix_conda_python(${target})
endif(APPLE)

# install the library
install(TARGETS ${target}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}
        )
endmacro()

#####################################################
## C++ libraries

# _common
add_python_module(_common "common/module-common.cc")

# _cpu
if (ENABLE_EMBREE AND ENABLE_TBB)

set(_cpu_sources
     cpu/module-cpu.cc
     cpu/Device.cc
     cpu/Scene.cc
     cpu/Geometry.cc
     cpu/GeometryPrism.cc
     cpu/GeometrySphere.cc
     cpu/Tracer.cc
     cpu/TracerDirect.cc
     cpu/Array.cc
   )

add_python_module(_cpu "${_cpu_sources}")
target_link_libraries(_cpu ${Embree_LIBRARIES} ${TBB_LIBRARIES})

endif()

# _gpu
if (ENABLE_CUDA AND ENABLE_OPTIX)

# Build the PTX files for the OptiX programs
set(_ptx_sources
	gpu/direct.cu
	gpu/GeometrySphere.cu
    gpu/GeometryPrism.cu)

CUDA_WRAP_SRCS(_ptx PTX generated_ptx_files ${_ptx_sources})
add_custom_target(ptx ALL
    DEPENDS ${generated_ptx_files} ${_ptx_sources}
    SOURCES ${_ptx_sources})

# Build the shared object
set(_gpu_sources
     gpu/module-gpu.cc
     gpu/Device.cc
     gpu/Scene.cc
     gpu/Geometry.cc
     gpu/GeometryPrism.cc
     gpu/GeometrySphere.cc
     gpu/Tracer.cc
     gpu/TracerDirect.cc
     gpu/Array.cc
   )

add_python_module(_gpu "${_gpu_sources}")
target_link_libraries(_gpu ${optix_LIBRARY})
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
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_python ALL DEPENDS ${files})
