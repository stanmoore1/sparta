# ##############################################################################
# Build the SPARTA HTML manual with Sphinx
#
# Adapted from cmake/Modules/Documentation.cmake in LAMMPS. The doxygen and
# globbed-toc steps are omitted: SPARTA has no programmer guide generated from
# source comments and no style-globbed toctrees.
#
# This is an alternative entry point to doc/Makefile, which remains the
# primary way to build the manual. Both drive the same sphinx-config.
# ##############################################################################

option(BUILD_DOC "Build SPARTA HTML documentation" OFF)

if(BUILD_DOC)
  option(BUILD_DOC_VENV "Build SPARTA documentation virtual environment" ON)
  mark_as_advanced(BUILD_DOC_VENV)

  # Current Sphinx versions require at least Python 3.8
  if(Python_VERSION VERSION_GREATER_EQUAL 3.8)
    set(Python3_EXECUTABLE ${Python_EXECUTABLE})
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(Python3_VERSION VERSION_LESS 3.8)
    message(
      FATAL_ERROR
        "Python 3.8 and up is required to build the SPARTA HTML documentation")
  endif()
  set(VIRTUALENV ${Python3_EXECUTABLE} -m venv)

  file(GLOB DOC_SOURCES CONFIGURE_DEPENDS ${SPARTA_DOC_DIR}/src/[^.]*.rst)

  set(SPHINX_CONFIG_DIR ${SPARTA_DOC_DIR}/utils/sphinx-config)
  set(SPHINX_CONFIG_FILE_TEMPLATE ${SPHINX_CONFIG_DIR}/conf.py.in)

  # the configuration is copied into the binary dir so that parallel builds
  # from one source tree do not collide over a shared generated conf.py
  set(DOC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc)
  set(DOC_BUILD_CONFIG_FILE ${DOC_BUILD_DIR}/conf.py)
  set(DOC_BUILD_STATIC_DIR ${DOC_BUILD_DIR}/_static)

  file(COPY ${SPHINX_CONFIG_DIR}/ DESTINATION ${DOC_BUILD_DIR})

  # conf.py.in uses @VAR@ placeholders, which configure_file substitutes
  # directly -- the same substitution doc/Makefile performs with sed. The
  # template names SPARTA_SOURCE_DIR/SPARTA_PYTHON_DIR, so map the tree's
  # SPARTA_SRC_DIR onto them here rather than diverging from the Makefile.
  set(SPARTA_SOURCE_DIR ${SPARTA_SRC_DIR})
  set(SPARTA_PYTHON_DIR ${SPARTA_CMAKE_DIR}/../python)
  configure_file(${SPHINX_CONFIG_FILE_TEMPLATE} ${DOC_BUILD_CONFIG_FILE} @ONLY)

  if(BUILD_DOC_VENV)
    add_custom_command(OUTPUT docenv COMMAND ${VIRTUALENV} docenv)

    set(DOCENV_BINARY_DIR ${CMAKE_BINARY_DIR}/docenv/bin)
    set(DOCENV_REQUIREMENTS_FILE ${SPARTA_DOC_DIR}/utils/requirements.txt)

    add_custom_command(
      OUTPUT ${DOC_BUILD_DIR}/requirements.txt
      DEPENDS docenv ${DOCENV_REQUIREMENTS_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${DOCENV_REQUIREMENTS_FILE}
              ${DOC_BUILD_DIR}/requirements.txt
      COMMAND ${DOCENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install --upgrade pip
      COMMAND ${DOCENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install --upgrade
              ${SPARTA_DOC_DIR}/utils/converters
      COMMAND ${DOCENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install -r
              ${DOC_BUILD_DIR}/requirements.txt --upgrade)

    set(DOCENV_DEPS docenv ${DOC_BUILD_DIR}/requirements.txt)
    if(NOT TARGET Sphinx::sphinx-build)
      add_executable(Sphinx::sphinx-build IMPORTED GLOBAL)
      set_target_properties(
        Sphinx::sphinx-build PROPERTIES IMPORTED_LOCATION
                                        "${DOCENV_BINARY_DIR}/sphinx-build")
    endif()
  else()
    find_package(Sphinx REQUIRED)
  endif()

  # MathJax is unpacked into _static so the built manual renders equations
  # with no network access, matching what doc/Makefile does with git clone.
  set(MATHJAX_URL "https://github.com/mathjax/MathJax/archive/4.1.3.tar.gz"
      CACHE STRING "URL for MathJax tarball")
  mark_as_advanced(MATHJAX_URL)

  if(NOT EXISTS ${DOC_BUILD_STATIC_DIR}/mathjax)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/mathjax.tar.gz)
      file(DOWNLOAD ${MATHJAX_URL} "${CMAKE_CURRENT_BINARY_DIR}/mathjax.tar.gz"
           STATUS DL_STATUS SHOW_PROGRESS)
      if(NOT DL_STATUS EQUAL 0)
        message(WARNING "Download of MathJax from ${MATHJAX_URL} failed. "
                        "Equations will not render without network access.")
      endif()
    endif()
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/mathjax.tar.gz)
      execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf mathjax.tar.gz
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      file(GLOB MATHJAX_VERSION_DIR CONFIGURE_DEPENDS
           ${CMAKE_CURRENT_BINARY_DIR}/MathJax-*)
      execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MATHJAX_VERSION_DIR}
                              ${DOC_BUILD_STATIC_DIR}/mathjax)
    endif()
  endif()

  add_custom_command(
    OUTPUT html
    DEPENDS ${DOC_SOURCES} ${DOCENV_DEPS} ${DOC_BUILD_CONFIG_FILE}
    COMMAND Sphinx::sphinx-build -b html -c ${DOC_BUILD_DIR} -d
            ${DOC_BUILD_DIR}/doctrees ${SPARTA_DOC_DIR}/src
            ${DOC_BUILD_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E create_symlink Manual.html
            ${DOC_BUILD_DIR}/html/index.html)

  add_custom_target(
    doc ALL
    DEPENDS html
    SOURCES ${SPARTA_DOC_DIR}/utils/requirements.txt ${DOC_SOURCES})

  # provides CMAKE_INSTALL_DOCDIR
  include(GNUInstallDirs)
  install(DIRECTORY ${DOC_BUILD_DIR}/html DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
