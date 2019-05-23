

foreach(obj ${OBJECTS})
    execute_process(COMMAND "${CMAKE_COMMAND}"
                    -E copy_if_different ${obj} ${OUTPUT_DIR}
                    )
    set(file_contents "${file_contents} \n${output}")
endforeach()
