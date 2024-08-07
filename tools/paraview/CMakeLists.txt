
set(PARAVIEW_TOOLS_PYTHON_FILES
    coprocessor.py
    grid2paraview_cells.py
    grid2paraview.py
    parallel_bucket_sort.py
    sort_sparta_grid_file.py
    surf2paraview.py)

set(PARAVIEW_TOOLS_SERIAL_PYTHON_TESTS
    grid2paraview_unit_test.py
    surf2paraview_unit_test.py)

set(PARAVIEW_TOOLS_PARALLEL_PYTHON_TESTS
    parallel_bucket_sort_unit_test.py)

set(PARAVIEW_TOOLS_TEST_FILES
    "test_data/grid.200"
    "test_data/circle_grid.txt"
    "test_data/data.circle_w_type"
    "test_data/data.sphere_w_type")

function(ADD_SERIAL_PARAVIEW_TESTS TEST_NAMES) 
    foreach(TEST_FILE IN LISTS TEST_NAMES)
        REMOVE_FILE_EXTENSION(TEST_NAME ${TEST_FILE})
        add_test(NAME "paraview_${TEST_NAME}" COMMAND
            ${PVPYTHON_EXECUTABLE} ${TEST_FILE})
    endforeach()
endfunction()

function(ADD_PARALLEL_PARAVIEW_TESTS TEST_NAMES NUM_PROCS)
    foreach(TEST_FILE IN LISTS TEST_NAMES)
        REMOVE_FILE_EXTENSION(TEST_NAME ${TEST_FILE})
        add_test(NAME "paraview_${TEST_NAME}_np_${NUM_PROCS}" COMMAND
            ${SPARTA_PARAVIEW_MPIEXEC} -n ${NUM_PROCS}
            ${PVBATCH_EXECUTABLE} --sym ${TEST_FILE})
    endforeach()
endfunction()

function(REMOVE_FILE_EXTENSION RESULT_NO_EXTENSION INPUT_WITH_EXTENSION)
    string(REGEX REPLACE "\\.[^.]*$" "" RESULT ${INPUT_WITH_EXTENSION})
    set(${RESULT_NO_EXTENSION} ${RESULT} PARENT_SCOPE)
endfunction()

if(SPARTA_ENABLE_TESTING AND SPARTA_ENABLE_PARAVIEW_TESTING)

   SET(PARAVIEW_TOOLS_ALL_FILES
       ${PARAVIEW_TOOLS_PYTHON_FILES}
       ${PARAVIEW_TOOLS_SERIAL_PYTHON_TESTS}
       ${PARAVIEW_TOOLS_PARALLEL_PYTHON_TESTS}
       ${PARAVIEW_TOOLS_TEST_FILES})

   set(PARAVIEW_TOOLS_ALL_FILES_SRC_DIR ${PARAVIEW_TOOLS_ALL_FILES})
   list(TRANSFORM PARAVIEW_TOOLS_ALL_FILES_SRC_DIR
       PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

   add_custom_target(paraview_python_files ALL DEPENDS
       ${PARAVIEW_TOOLS_ALL_FILES})
   add_custom_command(OUTPUT 
       ${PARAVIEW_TOOLS_ALL_FILES}
       COMMAND ${CMAKE_COMMAND} -E copy
           ${PARAVIEW_TOOLS_ALL_FILES_SRC_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}
       DEPENDS 
           ${PARAVIEW_TOOLS_ALL_FILES_SRC_DIR})

   ADD_SERIAL_PARAVIEW_TESTS("${PARAVIEW_TOOLS_SERIAL_PYTHON_TESTS}")
   ADD_PARALLEL_PARAVIEW_TESTS("${PARAVIEW_TOOLS_PARALLEL_PYTHON_TESTS}" 1)
   ADD_PARALLEL_PARAVIEW_TESTS("${PARAVIEW_TOOLS_PARALLEL_PYTHON_TESTS}" 4)

endif()
