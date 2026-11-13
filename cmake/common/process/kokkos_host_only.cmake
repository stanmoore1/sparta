# With the KOKKOS package for CUDA all sources of SPARTA are given to
# nvcc_wrapper (directly, or through kokkos_launch_compiler), but nvcc is very
# slow and most of them do not use Kokkos, so these only need the host
# compiler.  Sources that use Kokkos include it through a header with kokkos in
# its name, or through one of the generated style_*.h headers, which include the
# headers of the KOKKOS package styles.  A source that gets it through any other
# header fails to compile, and then that header has to be added to the regex.
#
# The sources compiled by the host compiler alone get SPARTA_KOKKOS_HOST_ONLY
# defined, so that accelerator_kokkos_defs.h does not include the Kokkos headers
# there but still sets SPARTA_KOKKOS_GPU, which changes the layout of structs
# that are shared with the KOKKOS package.
#
# The source file properties are scoped to a directory, so this has to be
# called from the CMakeLists.txt file that creates the target.  The sources of
# the KOKKOS package itself all need Kokkos, so that target is not passed here.

function(sparta_kokkos_host_only_sources target)
  if(NOT (PKG_KOKKOS AND Kokkos_ENABLE_CUDA))
    return()
  endif()
  get_filename_component(SPARTA_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
  if(NOT ((SPARTA_CXX_COMPILER_NAME STREQUAL "nvcc_wrapper")
          OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
    return()
  endif()

  get_target_property(SPARTA_TARGET_SOURCES ${target} SOURCES)
  get_target_property(SPARTA_TARGET_SOURCE_DIR ${target} SOURCE_DIR)
  foreach(SPARTA_SRC ${SPARTA_TARGET_SOURCES})
    if(NOT SPARTA_SRC MATCHES "\\.cpp$")
      continue()
    endif()
    if(IS_ABSOLUTE "${SPARTA_SRC}")
      set(SPARTA_SRC_PATH "${SPARTA_SRC}")
    else()
      set(SPARTA_SRC_PATH "${SPARTA_TARGET_SOURCE_DIR}/${SPARTA_SRC}")
    endif()
    file(STRINGS "${SPARTA_SRC_PATH}" KOKKOS_INCLUDES REGEX
         "^[ \t]*#[ \t]*include[ \t]*[<\"]([A-Za-z_/]*[Kk]okkos[A-Za-z_]*|style_[a-z_]+)\\.h")
    if(NOT KOKKOS_INCLUDES)
      set_property(SOURCE "${SPARTA_SRC}" APPEND PROPERTY COMPILE_OPTIONS "--host-only")
      set_property(SOURCE "${SPARTA_SRC}" APPEND PROPERTY COMPILE_DEFINITIONS SPARTA_KOKKOS_HOST_ONLY)
    endif()
  endforeach()
endfunction()
