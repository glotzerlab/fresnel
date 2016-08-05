################ C++ libraries
if (ENABLE_TRACE_CPU)

set(_cpu_sources
     cpu/module-cpu.cc
     cpu/Device.cc
     cpu/Scene.cc
   )

# TODO: determine if we need numpy or not with pybind11
# Need to define NO_IMPORT_ARRAY in every file but module.cc
# set_source_files_properties(${_cpu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

add_library (_cpu SHARED ${_cpu_sources})

# link the library to its dependencies
target_link_libraries(_cpu ${Embree_LIBRARIES} ${TBB_LIBRARIES})

# change the name of the library to be a valid python module
set_target_properties(_cpu PROPERTIES PREFIX "" OUTPUT_NAME "_cpu")

# fixup the library to be a working python module on macOS
if(APPLE)
    set_target_properties(_cpu PROPERTIES SUFFIX ".so")
    target_link_libraries(_cpu ${PYTHON_LIBRARY})
    fix_conda_python(_cpu)
endif(APPLE)

# install the library
install(TARGETS _cpu
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}
        )

endif(ENABLE_TRACE_CPU)

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy ${PACKAGE_NAME}/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_trace ALL DEPENDS ${files})
