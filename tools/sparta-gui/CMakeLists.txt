cmake_minimum_required(VERSION 3.20)

project(sparta-gui
  VERSION 1.0.0
  LANGUAGES CXX
  DESCRIPTION "The SPARTA GUI"
  HOMEPAGE_URL "https://sparta.github.io/sparta-gui/")

include(GNUInstallDirs)

##################################################################################
# configuration settings:
# prefer compiling plugin version only if standalone compile
option(SPARTA_GUI_USE_PLUGIN "Load SPARTA library dynamically at runtime" ON)
# interactive VTK 3D viewer; off by default because it needs an external VTK
# library (with Qt support).  When on, VTK is located with find_package; if it
# is not found the viewer is silently disabled so the build still succeeds.
option(SPARTA_GUI_USE_VTK "Build the interactive VTK 3D viewer (needs an external VTK library)" OFF)
# html documentation build is off by default
option(BUILD_DOC "Build SPARTA-GUI HTML documentation" OFF)
option(BUILD_DOC_ONLY "Build only the documentation, skip the application" OFF)
# testing support is off by default
option(ENABLE_TESTING "Build unit tests" OFF)
##################################################################################

# add automatic Qt infrastructure rules
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# same as SPARTA' default
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT BUILD_DOC_ONLY)
  # checks
  if (NOT SPARTA_SOURCE_DIR AND NOT SPARTA_GUI_USE_PLUGIN)
    message(FATAL_ERROR "Must provide -D SPARTA_SOURCE_DIR=<path to SPARTA sources> when not compiling with -D SPARTA_GUI_USE_PLUGIN=on")
  endif()
  set(SPARTA_HEADER_DIR ${SPARTA_SOURCE_DIR})
  if (NOT SPARTA_GUI_USE_PLUGIN AND NOT SPARTA_LIBRARY)
    message(FATAL_ERROR "Must provide -D SPARTA_LIBRARY=<path to SPARTA library file> when not compiling with -D SPARTA_GUI_USE_PLUGIN=on")
  endif()

  # install prefix, compiler and platform quirks, plugin loader, bundle resources
  include(${CMAKE_SOURCE_DIR}/cmake/Platform.cmake)

  # charts are rendered by the self-contained native QPainter backend; only the
  # LGPL Qt Widgets are needed (no Qt Charts / Graphs / Quick modules)
  # Qt Svg provides the SVG icon-engine plugin used to render the .svg icons;
  # declaring it here ensures the plugin is found and bundled by the deployment
  # tools (macdeployqt / windeployqt / flatpak)
  find_package(Qt6 6.2 REQUIRED COMPONENTS Gui Widgets Network Svg)
  message(STATUS "Using Qt version ${Qt6_VERSION}")

  # Qt-ADS (Advanced Docking System) provides the single-window docked panel
  # layout; vendored under thirdparty/qtads/ so the build has no network
  # dependency. Built as a static library and linked into sparta-gui below.
  # ADS links Qt6::GuiPrivate unconditionally on Linux, for
  # <qpa/qplatformnativeinterface.h>; that header ships in a separate
  # private-headers package on most distributions.
  if (UNIX AND NOT APPLE)
    get_target_property(_qtads_qt6gui_includedirs Qt6::Gui INTERFACE_INCLUDE_DIRECTORIES)
    find_file(QTADS_QPA_NATIVE_INTERFACE_H
      NAMES qpa/qplatformnativeinterface.h
      PATHS ${_qtads_qt6gui_includedirs}
      PATH_SUFFIXES "${Qt6_VERSION}/QtGui"
      NO_DEFAULT_PATH
    )
    if (NOT QTADS_QPA_NATIVE_INTERFACE_H)
      message(FATAL_ERROR "Qt-ADS needs Qt6 private headers "
        "(qpa/qplatformnativeinterface.h) to build on Linux. Install your "
        "distribution's Qt6 private-headers development package (e.g. "
        "qt6-base-private-dev on Debian/Ubuntu) and re-run cmake.")
    endif()
  endif()
  set(ADS_VERSION 4.4.1)
  set(BUILD_STATIC ON CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/qtads EXCLUDE_FROM_ALL)

  # the PROJECT_SOURCES file list, embedded resources, and the lepton_mini library
  include(${CMAKE_SOURCE_DIR}/cmake/Sources.cmake)

  # optional VTK 3D viewer: locate VTK (with Qt support) and, if present, add the
  # viewer source and remember to enable it below.  Missing VTK is not fatal --
  # the viewer is simply left out and its menu entries stay hidden at runtime.
  set(SPARTA_GUI_HAVE_VTK OFF)
  if (SPARTA_GUI_USE_VTK)
    # NB: we deliberately do NOT request the GUISupportQt module (QVTKOpenGLWidget).
    # On many distributions VTK's Qt integration is built against Qt5, which cannot
    # coexist with this Qt6 application in one process.  Instead the viewer renders
    # VTK off-screen and blits the frame into a plain Qt6 widget, so any system VTK
    # (Qt5-built, Qt6-built, or built with no Qt at all) works.
    find_package(VTK 9.0 QUIET COMPONENTS
      CommonCore CommonDataModel CommonColor CommonMath
      FiltersGeometry FiltersCore FiltersSources FiltersGeneral
      IOXML IOLegacy
      RenderingCore RenderingOpenGL2 RenderingAnnotation RenderingFreeType)
    if (VTK_FOUND)
      message(STATUS "Building the VTK 3D viewer with VTK ${VTK_VERSION}")
      set(SPARTA_GUI_HAVE_VTK ON)
      list(APPEND PROJECT_SOURCES
        ${CMAKE_SOURCE_DIR}/src/vtkviewer.cpp
        ${CMAKE_SOURCE_DIR}/src/vtkviewer.h
        ${CMAKE_SOURCE_DIR}/src/vtkfilters.cpp
        ${CMAKE_SOURCE_DIR}/src/vtkfilters.h)
    else()
      message(WARNING "SPARTA_GUI_USE_VTK=ON but no suitable VTK was found; "
        "the 3D viewer will be disabled. Install a VTK build with Qt support "
        "(e.g. libvtk9-qt-dev) or point CMake at it with -DVTK_DIR=<path>.")
    endif()
  endif()

  qt_add_executable(sparta-gui
    MANUAL_FINALIZATION
    ${MACOSX_ICON_FILE}
    ${PROJECT_SOURCES}
  )

  # when compiling for macOS request compilation of an app-bundle
  if(APPLE)
    string(TIMESTAMP CURRENT_YEAR "%Y")
    set_target_properties(sparta-gui PROPERTIES
      MACOSX_BUNDLE_INFO_PLIST ${MACOSX_PLIST_FILE}
      MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
      MACOSX_BUNDLE_ICON_FILE sparta-gui.icns
      MACOSX_BUNDLE_COPYRIGHT "(c) 2023 - ${CURRENT_YEAR}, Axel Kohlmeyer"
      MACOSX_BUNDLE TRUE
    )
    # purge app bundle directory during configure
    file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/sparta-gui.app)
  endif()
  install(TARGETS sparta-gui DESTINATION ${CMAKE_INSTALL_BINDIR})

  qt_finalize_executable(sparta-gui)

  # compilation settings
  if(SPARTA_GUI_USE_PLUGIN)
    target_compile_definitions(sparta-gui PRIVATE SPARTA_GUI_USE_PLUGIN)
    target_include_directories(sparta-gui PRIVATE ${SPARTA_PLUGINLIB_DIR})
  else()
    target_include_directories(sparta-gui PRIVATE ${SPARTA_HEADER_DIR})
    target_link_libraries(sparta-gui PRIVATE ${SPARTA_LIBRARY})
  endif()
  target_include_directories(sparta-gui PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(sparta-gui PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider)
  target_link_libraries(sparta-gui PRIVATE lepton_mini)
  target_compile_definitions(sparta-gui PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
  target_link_libraries(sparta-gui PRIVATE Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg)
  target_link_libraries(sparta-gui PRIVATE ads::qtadvanceddocking-qt6)

  # wire up the optional VTK 3D viewer: link the VTK modules, define the feature
  # macro the GUI code guards on, and run vtk_module_autoinit so VTK's OpenGL /
  # rendering object factories are registered (required or the render window
  # comes up blank).
  if(SPARTA_GUI_HAVE_VTK)
    target_compile_definitions(sparta-gui PRIVATE SPARTA_GUI_HAVE_VTK)
    target_link_libraries(sparta-gui PRIVATE ${VTK_LIBRARIES})
    vtk_module_autoinit(TARGETS sparta-gui MODULES ${VTK_LIBRARIES})
  endif()

  # unit tests and code coverage, only enabled on Linux
  include(${CMAKE_SOURCE_DIR}/cmake/Testing.cmake)
endif()

# optional instrumentation with a code sanitizer
include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)

# documentation build with Doxygen and Sphinx
include(${CMAKE_SOURCE_DIR}/cmake/Documentation.cmake)

# packaging targets (flatpak, dmg, Windows installer, tgz, source tarball)
include(${CMAKE_SOURCE_DIR}/cmake/Packaging.cmake)
