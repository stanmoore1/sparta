# CMakeLists.txt for SPARTA-GUI unit tests
# (only entered via add_subdirectory() from the top-level CMakeLists.txt)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# testing is already enabled via include(CTest) in the top-level CMakeLists.txt
include(GoogleTest)

# Find Qt6 (Qt5 is no longer supported)
find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets Test)

# check that the Xvfb runner and a supported screenshooter are present
find_program(XVFB_RUNNER
  NAMES xvfb-run
  REQUIRED
  DOC "Path to Xvfb runner script")

find_program(SHOOTER_BINARY
  NAMES magick import xfce4-screenshooter gnome-screenshot
  REQUIRED
  DOC "Path to a supported screenshooter program")
cmake_path(GET SHOOTER_BINARY FILENAME SHOOTER)
message(STATUS "Using ${SHOOTER} as screen shooter")

# configure virtual environment for tests using PyAutoGUI and Xvfb
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
set(TESTENV_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/testenv/bin/")
add_custom_command(
  OUTPUT ${TESTENV_BINARY_DIR}
  COMMAND ${Python3_EXECUTABLE} -m venv "${CMAKE_CURRENT_BINARY_DIR}/testenv"
  COMMAND ${TESTENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install --upgrade pip
  COMMAND ${TESTENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt --upgrade
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shooter.${SHOOTER} ${TESTENV_BINARY_DIR}/shooter
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
)
add_custom_target(setup_tests ALL
  ${CMAKE_COMMAND} -E echo "Setting up virtual environment for testing"
  DEPENDS ${TESTENV_BINARY_DIR}
  SOURCES requirements.txt shooter.magick shooter.import shooter.xfce4-screenshooter shooter.gnome-screenshot
)
set_target_properties(setup_tests PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/testenv)

# General command line tests on the SPARTA-GUI executable
add_test(NAME CommandLine.GetVersion
  COMMAND $<TARGET_FILE:sparta-gui> --platform offscreen -v)
set_tests_properties(CommandLine.GetVersion PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1"
  PASS_REGULAR_EXPRESSION "SPARTA-GUI \\(QT6\\) ${PROJECT_VERSION}")

add_test(NAME CommandLine.HasPlugin
  COMMAND $<TARGET_FILE:sparta-gui> --platform offscreen -h)
if (SPARTA_GUI_USE_PLUGIN)
  set_tests_properties(CommandLine.HasPlugin PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "Options:.?  -p, --pluginpath <path> +Set path to SPARTA")
else()
  set_tests_properties(CommandLine.HasPlugin PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "Options:.?  -h, --help   +Displays help on commandline options.")
endif()

# Exhaustive widget-tree walker.  Links only what the self-tests need; the
# real-window walks below drive the application's own dialogs.
# Links the whole application (minus its main()) so the walker can construct
# the real dialogs rather than stand-ins.  PROJECT_SOURCES already carries the
# generated resource file, which the dialogs need for their icons.
set(GUI_WALKER_SOURCES ${PROJECT_SOURCES})
list(FILTER GUI_WALKER_SOURCES EXCLUDE REGEX "/src/main\\.cpp$")

add_executable(test_gui_widgets
  test_gui_widgets.cpp
  ${GUI_WALKER_SOURCES}
)

# mirror the application target's include paths and definitions, or the shared
# sources compile differently here than they do in the shipped binary
target_include_directories(test_gui_widgets PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_gui_widgets PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")

target_link_libraries(test_gui_widgets PRIVATE
  GTest::gtest_main Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)

if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_gui_widgets PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_gui_widgets PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_gui_widgets MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_gui_widgets PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_gui_widgets PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

gtest_discover_tests(test_gui_widgets)

# Test executable for helpers
add_executable(test_helpers
  test_helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
)

target_include_directories(test_helpers PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_helpers PRIVATE GTest::gtest_main Qt6::Widgets)

# Discover tests for test executables
gtest_discover_tests(test_helpers)

# Test executable for StdCapture
add_executable(test_stdcapture
  test_stdcapture.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
)

target_include_directories(test_stdcapture PRIVATE  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_stdcapture PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_stdcapture)

# Test executable for FlagWarnings
add_executable(test_flagwarnings
  test_flagwarnings.cpp
  ${CMAKE_SOURCE_DIR}/src/flagwarnings.cpp
)

target_include_directories(test_flagwarnings PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_flagwarnings PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_flagwarnings)

# Test executable for the dump-image command builder
add_executable(test_dumpimage
  test_dumpimage.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimage.cpp
  ${CMAKE_SOURCE_DIR}/src/colormaps.cpp
)

target_include_directories(test_dumpimage PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dumpimage PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_dumpimage)

# Test executable for the displayed-image transform
add_executable(test_viewerdisplay
  test_viewerdisplay.cpp
  ${CMAKE_SOURCE_DIR}/src/viewerdisplay.cpp
)

target_include_directories(test_viewerdisplay PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_viewerdisplay PRIVATE GTest::gtest_main Qt6::Gui)

gtest_discover_tests(test_viewerdisplay)

# Test executable for the STL / SPARTA-surface import core
add_executable(test_stlimport
  test_stlimport.cpp
  ${CMAKE_SOURCE_DIR}/src/stlimport.cpp
)

target_include_directories(test_stlimport PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_stlimport PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_stlimport)

# Test executable for the ParaView-export command builders
add_executable(test_paraviewexport
  test_paraviewexport.cpp
  ${CMAKE_SOURCE_DIR}/src/paraviewexport.cpp
)

target_include_directories(test_paraviewexport PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_paraviewexport PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_paraviewexport)

# Test executable for the parametric-sweep expansion/reducers
add_executable(test_sweepspec
  test_sweepspec.cpp
  ${CMAKE_SOURCE_DIR}/src/sweepspec.cpp
)

target_include_directories(test_sweepspec PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sweepspec PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_sweepspec)

# Test executable for the run-provenance/report core
add_executable(test_runarchive
  test_runarchive.cpp
  ${CMAKE_SOURCE_DIR}/src/runarchive.cpp
)

target_include_directories(test_runarchive PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_runarchive PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_runarchive)

# Test executable for the editor snippet-library parser
add_executable(test_snippets
  test_snippets.cpp
  ${CMAKE_SOURCE_DIR}/src/snippets.cpp
)

target_include_directories(test_snippets PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_snippets PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_snippets)

# Test executable for the pre-run input-deck validator (Qt Core only)
add_executable(test_inputcheck
  test_inputcheck.cpp
  ${CMAKE_SOURCE_DIR}/src/inputcheck.cpp
)

target_include_directories(test_inputcheck PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_inputcheck PRIVATE GTest::gtest_main Qt6::Core)

gtest_discover_tests(test_inputcheck)

# Test executable for the case-setup model (Feature 8 canvas core)
add_executable(test_casemodel
  test_casemodel.cpp
  ${CMAKE_SOURCE_DIR}/src/casemodel.cpp
)

target_include_directories(test_casemodel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_casemodel PRIVATE GTest::gtest_main Qt6::Core)

gtest_discover_tests(test_casemodel)

# Test executable for the surface-quantity reduction core (Feature 3)
add_executable(test_surfreport
  test_surfreport.cpp
  ${CMAKE_SOURCE_DIR}/src/surfreport.cpp
)

target_include_directories(test_surfreport PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_surfreport PRIVATE GTest::gtest_main Qt6::Core)

gtest_discover_tests(test_surfreport)

# Test executable for the run-comparison core (Feature 5)
add_executable(test_runcompare
  test_runcompare.cpp
  ${CMAKE_SOURCE_DIR}/src/runcompare.cpp
  ${CMAKE_SOURCE_DIR}/src/runarchive.cpp
)

target_include_directories(test_runcompare PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_runcompare PRIVATE GTest::gtest_main Qt6::Core)

gtest_discover_tests(test_runcompare)

# Test executable for the VTK field-post-processing filters (Feature 9).
# Only built when a suitable VTK is available; the filters need no display.
find_package(VTK 9.0 QUIET COMPONENTS
  CommonCore CommonDataModel FiltersCore FiltersGeneral FiltersSources)
if (VTK_FOUND)
  add_executable(test_vtkfilters
    test_vtkfilters.cpp
    ${CMAKE_SOURCE_DIR}/src/vtkfilters.cpp
  )
  target_include_directories(test_vtkfilters PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(test_vtkfilters PRIVATE GTest::gtest_main Qt6::Core ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_vtkfilters MODULES ${VTK_LIBRARIES})
  gtest_discover_tests(test_vtkfilters)

  # The find_package above narrowed VTK_LIBRARIES to the five modules the pure
  # filter tests need. The scene renders, so ask again for the full set the
  # application itself links -- after test_vtkfilters has already been linked,
  # so that target keeps its smaller dependency.
  find_package(VTK 9.0 QUIET COMPONENTS
    CommonCore CommonDataModel CommonColor CommonMath
    FiltersGeometry FiltersCore FiltersSources FiltersGeneral
    IOXML IOLegacy
    RenderingCore RenderingOpenGL2 RenderingAnnotation RenderingFreeType)

  # The 3D scene widget and its standalone window. Links the application
  # sources the same way the widget walker does: the scene reaches into the
  # chart window for its line probe, and hand-listing that dependency chain
  # drifts out of date the moment anything moves.
  set(VTK_SCENE_SOURCES ${PROJECT_SOURCES})
  list(FILTER VTK_SCENE_SOURCES EXCLUDE REGEX "/src/main\\.cpp$")

  add_executable(test_vtkscene
    test_vtkscene.cpp
    ${VTK_SCENE_SOURCES}
  )
  target_include_directories(test_vtkscene PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
  )
  target_compile_definitions(test_vtkscene PRIVATE
    SPARTA_GUI_VERSION="${PROJECT_VERSION}" SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_vtkscene PRIVATE
    GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg
    lepton_mini ads::qtadvanceddocking-qt6 ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_vtkscene MODULES ${VTK_LIBRARIES})
  if (SPARTA_GUI_USE_PLUGIN)
    target_compile_definitions(test_vtkscene PRIVATE SPARTA_GUI_USE_PLUGIN)
    target_include_directories(test_vtkscene PRIVATE ${SPARTA_PLUGINLIB_DIR})
  endif()
  gtest_discover_tests(test_vtkscene)
endif()

# Test executable for the movie probing and frame selection helpers
add_executable(test_movieimport
  test_movieimport.cpp
  ${CMAKE_SOURCE_DIR}/src/movieimport.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
)

target_include_directories(test_movieimport PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_movieimport PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_movieimport)

# Test executable for the converted image cache
add_executable(test_imagecache
  test_imagecache.cpp
  ${CMAKE_SOURCE_DIR}/src/imagecache.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
)

target_include_directories(test_imagecache PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_imagecache PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_imagecache)

# Test executable for the least-squares / linear-algebra toolkit (Qt-free)
add_executable(test_leastsquares
  test_leastsquares.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
)

target_include_directories(test_leastsquares PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_leastsquares PRIVATE GTest::gtest_main)

gtest_discover_tests(test_leastsquares)

# Test executable for the PlotData model and file parsers
add_executable(test_plotdata
  test_plotdata.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdata.cpp
)

target_include_directories(test_plotdata PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_plotdata PRIVATE GTest::gtest_main Qt6::Core)

gtest_discover_tests(test_plotdata)

# Test executable for the vendored LeptonMini expression parser (Qt-free)
add_executable(test_lepton
  test_lepton.cpp
)

target_link_libraries(test_lepton PRIVATE GTest::gtest_main lepton_mini)

gtest_discover_tests(test_lepton)

# Test executable for the Levenberg-Marquardt solver (Qt-free; uses leastsquares)
add_executable(test_levmar
  test_levmar.cpp
  ${CMAKE_SOURCE_DIR}/src/levmar.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
)

target_include_directories(test_levmar PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_levmar PRIVATE GTest::gtest_main)

gtest_discover_tests(test_levmar)

# Test executable for the custom-function evaluator and nonlinear fit
# (LeptonMini + the Levenberg-Marquardt / leastsquares core + Qt Core)
add_executable(test_customfunc
  test_customfunc.cpp
  ${CMAKE_SOURCE_DIR}/src/customfunc.cpp
  ${CMAKE_SOURCE_DIR}/src/levmar.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
)

target_include_directories(test_customfunc PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_customfunc PRIVATE GTest::gtest_main Qt6::Core lepton_mini)

gtest_discover_tests(test_customfunc)

# Test executable for the post-processing analyses (Qt-free)
add_executable(test_analysis
  test_analysis.cpp
  ${CMAKE_SOURCE_DIR}/src/analysis.cpp
)

target_include_directories(test_analysis PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_analysis PRIVATE GTest::gtest_main)

gtest_discover_tests(test_analysis)

# Test executable for the axis-layout helpers (Qt-free)
add_executable(test_plotaxismath
  test_plotaxismath.cpp
  ${CMAKE_SOURCE_DIR}/src/plotaxismath.cpp
)

target_include_directories(test_plotaxismath PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_plotaxismath PRIVATE GTest::gtest_main)

gtest_discover_tests(test_plotaxismath)

# Test executable for the curve fits (Qt-free; depends on the leastsquares core)
add_executable(test_fitting
  test_fitting.cpp
  ${CMAKE_SOURCE_DIR}/src/fitting.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
)

target_include_directories(test_fitting PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_fitting PRIVATE GTest::gtest_main)

gtest_discover_tests(test_fitting)

# only run framebuffer tests without sanitizers
if(ENABLE_SANITIZER STREQUAL "none")
########################################################################
# Tests that check the GUI environment and the GUI
# IMPORTANT NOTE: we must use a different display number for the -n flag
# for each XVFB_RUNNER invocation, to allow parallel execution of tests.

# set up environment for python testing
set(PYTHON_TEST_ENVIRONMENT PYTHONUNBUFFERED=1)
list(APPEND PYTHON_TEST_ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
list(APPEND PYTHON_TEST_ENVIRONMENT "OMP_NUM_THREADS=1")
list(APPEND PYTHON_TEST_ENVIRONMENT "SPARTA_GUI=$<TARGET_FILE:sparta-gui>")
list(APPEND PYTHON_TEST_ENVIRONMENT "PATH=${TESTENV_BINARY_DIR}:$ENV{PATH}")

# test screen shooter wrapper script
add_test(NAME Framebuffer.CreateScreenshot
  COMMAND ${XVFB_RUNNER} -n 11 -s "-screen 0 1024x768x24" -w 1
  ${TESTENV_BINARY_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR}/test_shooter.py
)

# test if PyAutoGUI is usable
add_test(NAME Framebuffer.CheckSize
  COMMAND ${XVFB_RUNNER} -n 12 -s "-screen 0 1024x768x24" -w 1
  ${TESTENV_BINARY_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR}/test_xvfbsize.py
)
set_tests_properties(Framebuffer.CreateScreenshot  Framebuffer.CheckSize
                     PROPERTIES ENVIRONMENT "${PYTHON_TEST_ENVIRONMENT}")

# test basic SPARTA-GUI editor tasks
add_test(NAME Framebuffer.GUIEditorChecks
  COMMAND ${XVFB_RUNNER} -n 13 -s "-screen 0 1024x768x24" -w 1
  ${TESTENV_BINARY_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_edit.py
)
set_tests_properties(Framebuffer.GUIEditorChecks
                     PROPERTIES ENVIRONMENT "${PYTHON_TEST_ENVIRONMENT}")

# Visual-effect and end-to-end feature suites.  Unlike the tests above these
# manage their own Xvfb (and a window manager, without which nothing ever gets
# keyboard focus and every step after the first silently does nothing), and
# they need no PyAutoGUI, so they run under the plain interpreter.
find_program(XDOTOOL_BINARY xdotool DOC "Path to xdotool")
find_program(WM_BINARY NAMES openbox matchbox-window-manager fluxbox
             DOC "Path to a minimal window manager")

if (XDOTOOL_BINARY AND WM_BINARY AND SHOOTER_BINARY)
  set(GUI_DRIVE_ENVIRONMENT
    "SPARTA_GUI=$<TARGET_FILE:sparta-gui>"
    "SPARTA_LIB_DIR=${CMAKE_BINARY_DIR}"
    "SPARTA_FIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  )

  add_test(NAME GuiDrive.VisualEffects
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_visual.py)
  set_tests_properties(GuiDrive.VisualEffects PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/visual-shots"
    TIMEOUT 600)

  add_test(NAME GuiDrive.Features
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_features.py)
  set_tests_properties(GuiDrive.Features PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/feature-shots"
    TIMEOUT 900)
else()
  message(STATUS "Skipping the visual and feature GUI suites: "
    "they need xdotool, a window manager (e.g. openbox) and a screenshooter")
endif()
endif()
