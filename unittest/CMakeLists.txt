# Standalone C++ unit tests for self-contained SPARTA kernels.
#
# These tests link only the handful of SPARTA source files they exercise
# (math_extra, random_knuth), so they build quickly and do not require MPI,
# KOKKOS, or the full SPARTA library.  Enable with -DSPARTA_ENABLE_UNIT_TESTING=ON.

add_executable(sparta_unittest_math_rng
  ${CMAKE_CURRENT_SOURCE_DIR}/test_math_rng.cpp
  ${SPARTA_SRC_DIR}/math_extra.cpp
  ${SPARTA_SRC_DIR}/random_knuth.cpp)

target_include_directories(sparta_unittest_math_rng PRIVATE ${SPARTA_SRC_DIR})

add_test(NAME unit.math_rng COMMAND sparta_unittest_math_rng)
set_tests_properties(unit.math_rng PROPERTIES
  PASS_REGULAR_EXPRESSION "All unit tests passed"
  FAIL_REGULAR_EXPRESSION "FAIL")
