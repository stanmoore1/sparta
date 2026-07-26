##########################################################################
# Unit tests (GoogleTest via CTest) and code coverage reporting.
# Testing is only supported on native Linux builds.  Expects the
# sparta-gui target to exist (the coverage flags are applied to it).
##########################################################################

if(ENABLE_TESTING AND (NOT CMAKE_CROSSCOMPILING) AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  message(STATUS "Testing is enabled")
  include(CTest)

  # Compiler specific features for testing
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    option(ENABLE_COVERAGE "Enable collecting code coverage data" OFF)
    if(ENABLE_COVERAGE)
      target_compile_options(sparta-gui PUBLIC --coverage)
      target_link_options(sparta-gui PUBLIC --coverage)

      # ...and everything created from here down, which is where the test
      # binaries are. Instrumenting the sparta-gui target alone covered the
      # application binary only: the test executables compile their own copies
      # of the sources rather than linking that target (it is an executable,
      # nothing can link it), so no .gcda was ever written for them and the
      # report reflected only the few suites that drive the built application.
      # These have to be set before add_subdirectory(test) to reach it.
      add_compile_options(--coverage)
      add_link_options(--coverage)
    endif()
  endif()

  add_subdirectory(test)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(ENABLE_COVERAGE)
      find_program(GCOVR_BINARY gcovr)
      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(GCOVR DEFAULT_MSG GCOVR_BINARY)
      if(GCOVR_FOUND)
        message(STATUS "Code coverage reporting with gcovr is enabled")
        get_filename_component(ABSOLUTE_SOURCE_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)

        set(COVERAGE_HTML_DIR ${CMAKE_BINARY_DIR}/coverage_html)
        add_custom_target(coverage_html_folder
          COMMAND ${CMAKE_COMMAND} -E make_directory ${COVERAGE_HTML_DIR})

        # The excludes keep the report about this project: with the test
        # binaries instrumented too, GoogleTest and the vendored dependencies
        # would otherwise dominate it and bury the numbers that matter.
        #
        # VERBATIM, or the generator hands the exclude patterns to a shell that
        # expands ".*" as a glob before gcovr sees it -- ".../test/.*" arrives
        # as ".../test/." plus a stray ".../test/..", which gcovr rejects as an
        # unrecognized argument.  The target failed that way every time it was
        # run, which is why these numbers had never been measured.  The double
        # quotes in the source are CMake's, and are gone by then.
        add_custom_target(
          coverage
          COMMAND ${GCOVR_BINARY} -s --html --html-nested --html-self-contained
                  -r ${ABSOLUTE_SOURCE_DIR} --object-directory=${CMAKE_BINARY_DIR}
                  --exclude "${ABSOLUTE_SOURCE_DIR}/thirdparty/.*"
                  --exclude "${ABSOLUTE_SOURCE_DIR}/test/.*"
                  --exclude ".*/_deps/.*"
                  --exclude ".*/moc_.*"
                  --exclude ".*/qrc_.*"
                  -o ${COVERAGE_HTML_DIR}/index.html
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
          VERBATIM
          COMMENT "Generating HTML coverage report..."
        )
        add_dependencies(coverage coverage_html_folder)

        add_custom_target(clean_coverage
          ${CMAKE_COMMAND} -E remove_directory ${COVERAGE_HTML_DIR}
          COMMAND ${CMAKE_COMMAND} -E remove -f */*.gcda */*/*.gcda */*/*/*.gcda
          */*/*/*/*.gcda */*/*/*/*/*.gcda */*/*/*/*/*/*.gcda
          */*/*/*/*/*/*/*.gcda */*/*/*/*/*/*/*/*.gcda
          */*/*/*/*/*/*/*/*/*.gcda */*/*/*/*/*/*/*/*/*/*.gcda
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
          COMMENT "Deleting coverage report and coverage data files..."
        )
      endif()
    endif()
  endif()
else()
  message(STATUS "Testing is disabled")
endif()
