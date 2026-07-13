# Register the SPARTA Python/library API test (python/test_library.py).
#
# The test drives SPARTA through its C library interface via the ctypes
# wrapper in python/sparta.py, so it requires:
#   - SPARTA_ENABLE_TESTING = ON
#   - PKG_PYTHON            = ON   (library interface compiled in)
#   - BUILD_SHARED_LIBS     = ON   (libsparta.so loadable by ctypes)
#   - a Python 3 interpreter
#
# If testing is enabled but these prerequisites are not met, the test is
# skipped with a status message rather than failing configuration, so the
# existing static-library CI jobs are unaffected.

if(SPARTA_ENABLE_TESTING AND PKG_PYTHON)
  if(BUILD_SHARED_LIBS)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_Interpreter_FOUND)
      add_test(
        NAME python.library
        COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_library.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python)
      # let ctypes find the freshly built libsparta.so
      set_tests_properties(python.library PROPERTIES
        ENVIRONMENT
          "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:${TARGET_SPARTA_LIB}>:$ENV{LD_LIBRARY_PATH}"
        PASS_REGULAR_EXPRESSION "All library API tests passed"
        FAIL_REGULAR_EXPRESSION "FAIL")
    else()
      message(STATUS
        "Python interpreter not found; skipping python.library test")
    endif()
  else()
    message(STATUS
      "BUILD_SHARED_LIBS is OFF; skipping python.library test "
      "(libsparta shared library required by the ctypes wrapper)")
  endif()
endif()
