# Create tests that contain each header separately. We do not  run these tests
# but we just try to compile them.

# Globbing all the header filenames to test for self-containment and presence of header guards
set(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")
file(GLOB KOKKOS_CORE_HEADERS RELATIVE ${BASE_DIR}/core/src ${BASE_DIR}/core/src/*.hpp ${BASE_DIR}/core/src/*.h)
file(GLOB KOKKOS_CONTAINERS_HEADERS RELATIVE ${BASE_DIR}/containers/src ${BASE_DIR}/containers/src/*.hpp)
file(GLOB KOKKOS_ALGORITHMS_HEADERS RELATIVE ${BASE_DIR}/algorithms/src ${BASE_DIR}/algorithms/src/*.hpp)

# erroring out when deprecated code is disabled and raising warnings that are treated as errors in the CI otherwise
if(NOT Kokkos_ENABLE_DEPRECATED_CODE_4 OR Kokkos_ENABLE_DEPRECATION_WARNINGS)
  list(REMOVE_ITEM KOKKOS_CONTAINERS_HEADERS "Kokkos_Vector.hpp" "Kokkos_StaticCrsGraph.hpp")
  list(REMOVE_ITEM KOKKOS_CORE_HEADERS "Kokkos_Future.hpp" "Kokkos_TaskScheduler.hpp")
endif()

# FIXME circular dependencies and execution space type not complete
list(REMOVE_ITEM KOKKOS_CORE_HEADERS "Kokkos_MemoryPool.hpp")

foreach(_header ${KOKKOS_CORE_HEADERS} ${KOKKOS_CONTAINERS_HEADERS} ${KOKKOS_ALGORITHMS_HEADERS})
  string(REGEX REPLACE "[\./]" "_" header_test_name ${_header})
  set(header_test_name Kokkos_HeaderSelfContained_${header_test_name})
  set_source_files_properties(tstHeader.cpp PROPERTIES LANGUAGE ${KOKKOS_COMPILE_LANGUAGE})
  add_executable(${header_test_name} tstHeader.cpp)
  target_link_libraries(${header_test_name} PRIVATE Kokkos::kokkos)
  target_compile_definitions(${header_test_name} PRIVATE KOKKOS_HEADER_TEST_NAME=${_header})
endforeach()
