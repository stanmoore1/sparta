# Copied from
# https://github.com/lammps/lammps/tree/3b77546eb9b0b5f7a402531b130e3b6dfad98a0c
# - Find fftw3 Find the native double precision FFTW3 headers and libraries.
#
# FFTW3_INCLUDE_DIRS  - where to find fftw3.h, etc. FFTW3_LIBRARIES     - List
# of libraries when using fftw3. FFTW3_OMP_LIBRARIES - List of libraries when
# using fftw3. FFTW3_FOUND         - True if fftw3 found.
#

# with FFT_SINGLE the single precision fftw3f library is used, whose
# fftwf_* API the code calls when FFT_SCALAR is float

if(FFT_SINGLE)
  set(FFTW3_PREC_SUFFIX "f")
else()
  set(FFTW3_PREC_SUFFIX "")
endif()

find_package(PkgConfig)

pkg_check_modules(PC_FFTW3 fftw3${FFTW3_PREC_SUFFIX})
find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS ${PC_FFTW3_INCLUDE_DIRS})
find_library(
  FFTW3_LIBRARY
  NAMES fftw3${FFTW3_PREC_SUFFIX}
  HINTS ${PC_FFTW3_LIBRARY_DIRS})
find_library(
  FFTW3_OMP_LIBRARY
  NAMES fftw3${FFTW3_PREC_SUFFIX}_omp
  HINTS ${PC_FFTW3_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FFTW3_FOUND to TRUE if all
# listed variables are TRUE

find_package_handle_standard_args(FFTW3 DEFAULT_MSG FFTW3_LIBRARY
                                  FFTW3_INCLUDE_DIR)

# Copy the results to the output variables and target.
if(FFTW3_FOUND)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})

  if(NOT TARGET FFTW3::FFTW3)
    add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
    set_target_properties(
      FFTW3::FFTW3
      PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION
                                                       "${FFTW3_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}")
  endif()
  if(FFTW3_OMP_LIBRARY)
    set(FFTW3_OMP_LIBRARIES ${FFTW3_OMP_LIBRARY})
    if(NOT TARGET FFTW3::FFTW3_OMP)
      add_library(FFTW3::FFTW3_OMP UNKNOWN IMPORTED)
      set_target_properties(
        FFTW3::FFTW3_OMP
        PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION
                                                         "${FFTW3_OMP_LIBRARY}"
                   INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}")
    endif()
  endif()
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY FFTW3_OMP_LIBRARY)
