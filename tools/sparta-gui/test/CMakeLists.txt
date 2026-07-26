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
find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets Test PrintSupport)

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

# PyAutoGUI needs tkinter, and tkinter is a compiled stdlib module that a
# locally built interpreter (or a slim container image) often does not have.
# A venv inherits the stdlib of whatever built it, so picking the wrong
# interpreter here fails at import time in every PyAutoGUI test with a message
# about installing python3-tk -- three test failures whose cause is nowhere
# near the tests. Probe instead, and say plainly when none of them will do.
set(TESTENV_PYTHON "")
foreach(_cand IN ITEMS "${Python3_EXECUTABLE}" python3 python3.13 python3.12 python3.11 python3.10)
  if(TESTENV_PYTHON)
    continue()
  endif()
  find_program(_cand_path NAMES "${_cand}")
  if(_cand_path)
    execute_process(COMMAND "${_cand_path}" -c "import tkinter"
      RESULT_VARIABLE _tk_rc OUTPUT_QUIET ERROR_QUIET)
    if(_tk_rc EQUAL 0)
      set(TESTENV_PYTHON "${_cand_path}")
    endif()
  endif()
  unset(_cand_path CACHE)
endforeach()
if(TESTENV_PYTHON)
  message(STATUS "PyAutoGUI test environment will use ${TESTENV_PYTHON}")
else()
  set(TESTENV_PYTHON "${Python3_EXECUTABLE}")
  message(WARNING "No Python with tkinter found: the PyAutoGUI tests "
    "(Framebuffer.*) will fail on import. Install the tkinter package for "
    "${Python3_EXECUTABLE} (on Debian/Ubuntu: python3-tk).")
endif()

set(TESTENV_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/testenv/bin/")
add_custom_command(
  OUTPUT ${TESTENV_BINARY_DIR}
  COMMAND ${TESTENV_PYTHON} -m venv "${CMAKE_CURRENT_BINARY_DIR}/testenv"
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

# The walker needs two things from outside the build to do its whole job, and
# it skips the tests that need them rather than failing.  Both went unset in
# every configure, so seven walks -- among them the image viewer with its eight
# settings tabs, by far the largest window in the application -- were quietly
# skipped everywhere instead of running.
#
#  SPARTA_FIXTURES   the STL meshes and the in.surfq deck; always known
#  SPARTA_PLUGIN_LIB a shared libsparta for the windows that read species,
#                    surfaces and box extents out of a live instance
set(SPARTA_TEST_LIBRARY "" CACHE FILEPATH
  "Shared libsparta used by the live-window tests (empty: probe the usual places)")
if (SPARTA_TEST_LIBRARY)
  set(_spa_test_lib "${SPARTA_TEST_LIBRARY}")
elseif (SPARTA_LIBRARY)
  # linked (non-plugin) builds already point at one
  set(_spa_test_lib "${SPARTA_LIBRARY}")
else()
  # a sibling build tree of the SPARTA checkout this GUI lives in
  file(GLOB _spa_candidates
    "${CMAKE_SOURCE_DIR}/../../build*/src/libsparta*.so.*"
    "${CMAKE_SOURCE_DIR}/../../build*/libsparta*.so.*")
  if (_spa_candidates)
    list(GET _spa_candidates 0 _spa_test_lib)
  else()
    set(_spa_test_lib "")
  endif()
endif()

if (NOT _spa_test_lib OR NOT EXISTS "${_spa_test_lib}")
  set(_spa_test_lib "")
  message(STATUS "No shared libsparta found: the live-window tests will skip. "
    "Point -D SPARTA_TEST_LIBRARY=... at one to run them.")
else()
  message(STATUS "Live-window tests will use ${_spa_test_lib}")
endif()

# Compiled in rather than passed through the ctest environment: two variables
# is a two-element list, gtest_discover_tests writes the elements out as
# separate arguments, and set_tests_properties then reads the second one as
# the name of another property and drops the value.  Baking the paths in also
# means the binary does the same thing when run by hand, which is how anyone
# looking at a walker failure runs it.  Both are still overridable by the
# environment.
target_compile_definitions(test_gui_widgets PRIVATE
  SPARTA_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")

gtest_discover_tests(test_gui_widgets)

# The main window.  Same source list as the walker -- the whole application
# minus its main() -- because this constructs the real SpartaGui rather than a
# stand-in.  Nothing did that before, so the largest source file in the tree
# had its menus, shortcuts and workspace layouts checked only by driving the
# running program under Xvfb.  This runs offscreen in about a second.
add_executable(test_mainwindow
  test_mainwindow.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_mainwindow PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_mainwindow PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  # The window's constructor loops on a modal until it has a library to load,
  # so without one this suite would hang rather than fail. It skips instead.
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_mainwindow PRIVATE
  GTest::gtest_main Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)

if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_mainwindow PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_mainwindow PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_mainwindow MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_mainwindow PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_mainwindow PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# One ctest entry rather than gtest_discover_tests, under Xvfb: the workspaces
# that show the viewer build a VTK scene, and VTK's "off-screen" rendering is
# still vtkXOpenGLRenderWindow speaking GLX, so it needs an X server even though
# Qt itself is on the offscreen platform.  TEST_LAUNCHER, which would let the
# discovered per-case commands be wrapped, is CMake 3.29 and this targets 3.20.
if (SPARTA_GUI_HAVE_VTK)
  add_test(NAME MainWindow.All COMMAND ${XVFB_RUNNER} -a -s "-screen 0 1024x768x24" -w 1
    $<TARGET_FILE:test_mainwindow>)
  # RESOURCE_LOCK, because -a still races: two concurrent xvfb-run invocations
  # can both decide the same display is free before either starts a server on
  # it. The loser's cases all pass and xvfb-run exits non-zero anyway, so the
  # suite is reported as failed with nothing in its output to say why.
  set_tests_properties(MainWindow.All PROPERTIES TIMEOUT 300 RESOURCE_LOCK xvfb)
else()
  gtest_discover_tests(test_mainwindow)
endif()

# The bundled lookup tables: the help index, the image-style list, the syntax
# table and the internal-command list. Needs the generated resource file (the
# tables are read from the bundle, not from disk, which is how the application
# reads them) and the library wrapper, to ask a real SPARTA which computes and
# fixes it actually has rather than trusting the list.
qt6_add_resources(TEST_RESOURCE_SOURCES ${CMAKE_SOURCE_DIR}/resources/spartagui.qrc)
add_executable(test_resources
  test_resources.cpp
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
  ${PLUGIN_LOADER_SRC}
  ${TEST_RESOURCE_SOURCES}
)
target_include_directories(test_resources PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_resources PRIVATE
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}"
  SPARTA_DOC_DIR="${CMAKE_SOURCE_DIR}/../../doc")
target_link_libraries(test_resources PRIVATE GTest::gtest_main Qt6::Widgets)
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_resources PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_resources PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

gtest_discover_tests(test_resources)

# What the renderer draws. The four dump image features back-ported into the
# SPARTA core have GUI controls and emitted-string tests; nothing ever ran the
# C++ that has to make the picture different, so a keyword the parser accepts
# and then ignores would pass every check that existed.
add_executable(test_render
  test_render.cpp
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
  ${PLUGIN_LOADER_SRC}
)
target_include_directories(test_render PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_render PRIVATE
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}"
  SPARTA_RENDER_FIXTURES="${CMAKE_SOURCE_DIR}/../../examples/circle")
target_link_libraries(test_render PRIVATE GTest::gtest_main Qt6::Widgets)
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_render PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_render PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

gtest_discover_tests(test_render)

# The chart renderer. It draws every chart the application shows and had no
# test at all; a setter that stores its argument and a renderer that ignores it
# look identical from outside, so these render into an image and ask the pixels.
add_executable(test_plotwidget
  test_plotwidget.cpp
  ${CMAKE_SOURCE_DIR}/src/plotwidget.cpp
  ${CMAKE_SOURCE_DIR}/src/plotaxismath.cpp
)
target_include_directories(test_plotwidget PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_plotwidget PRIVATE GTest::gtest_main Qt6::Widgets)
gtest_discover_tests(test_plotwidget)

# The parametric sweep's results table: the only table filled in while runs are
# happening, one row at a time, from a driver that can be stopped part way. A
# model that miscounts or forgets to bracket an insertion corrupts the view
# rather than failing, so QAbstractItemModelTester watches it too.
# Links the whole application: sweeppanel.cpp also holds SweepController, which
# drives the main window and the chart panel, so the model cannot be compiled
# on its own.
add_executable(test_sweeppanel
  test_sweeppanel.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_sweeppanel PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_sweeppanel PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_sweeppanel PRIVATE
  GTest::gtest_main Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_sweeppanel PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_sweeppanel PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_sweeppanel MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_sweeppanel PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_sweeppanel PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_sweeppanel)

# The welcome screen: the first thing a new user sees, and untested. Needs the
# resource bundle, because the gallery only offers decks it has a shipped
# thumbnail for.
add_executable(test_welcomescreen
  test_welcomescreen.cpp
  ${CMAKE_SOURCE_DIR}/src/welcomescreen.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${TEST_RESOURCE_SOURCES}
)
target_include_directories(test_welcomescreen PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_welcomescreen PRIVATE
  GTest::gtest_main Qt6::Widgets Qt6::Svg Qt6::Test)
gtest_discover_tests(test_welcomescreen)

# The column chooser behind File > Plot Data File. Everything it returns is an
# index into the data it hands back, so an off-by-one between the two draws a
# chart of the wrong quantity without complaint.
add_executable(test_plotdatadialog
  test_plotdatadialog.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdatadialog.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdata.cpp
  ${CMAKE_SOURCE_DIR}/src/customfunc.cpp
  ${CMAKE_SOURCE_DIR}/src/levmar.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
)
target_include_directories(test_plotdatadialog PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton/include)
target_link_libraries(test_plotdatadialog PRIVATE
  GTest::gtest_main Qt6::Widgets Qt6::Test lepton_mini)
gtest_discover_tests(test_plotdatadialog)

# The light/dark appearance: applied once in main() before the main window
# exists, so anything wrong with it is wrong for the whole session. Needs the
# resource bundle for :/style.qss.
add_executable(test_theme
  test_theme.cpp
  ${CMAKE_SOURCE_DIR}/src/theme.cpp
  ${TEST_RESOURCE_SOURCES}
)
target_include_directories(test_theme PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_theme PRIVATE GTest::gtest_main Qt6::Widgets Qt6::Svg)
gtest_discover_tests(test_theme)

# The Output panel. Its warning/error badge is, on a run of thousands of lines,
# the only thing telling the user whether anything went wrong -- and it had no
# test. The panel's Quit/Stop/Run shortcuts call straight into the main window,
# so this links the whole application rather than stubbing three methods.
add_executable(test_logwindow
  test_logwindow.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_logwindow PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_logwindow PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_logwindow PRIVATE
  GTest::gtest_main Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_logwindow PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_logwindow PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_logwindow MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_logwindow PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_logwindow PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_logwindow)

# The dump-image settings dialog: eight tabs and a hundred-odd controls that
# decide what every rendered picture looks like. Until the dialog was split out
# of ImageViewer::settingsDialog() it could not be built without a running
# simulator, so the widget-to-struct mapping -- the part where a control wired
# to the wrong field silently renders the wrong picture -- had never been
# checked. Four leaf sources, no SPARTA library, no display.
add_executable(test_dumpimagesettings
  test_dumpimagesettings.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimagesettingsdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimage.cpp
  ${CMAKE_SOURCE_DIR}/src/colormaps.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
)
target_include_directories(test_dumpimagesettings PRIVATE ${CMAKE_SOURCE_DIR}/src)
# GTest::gtest, not gtest_main: the test brings its own main() so it can put Qt
# on the offscreen platform before the QApplication exists.
target_link_libraries(test_dumpimagesettings PRIVATE
  GTest::gtest Qt6::Widgets Qt6::Test)
gtest_discover_tests(test_dumpimagesettings)

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

# Test executable for the displayed-image transform. Deliberately links only
# Qt6::Gui: the transform is pure, and keeping it in its own translation unit
# apart from the widget that uses it is what lets this stay a fast unit test
# rather than one that drags in the whole widget stack.
add_executable(test_displaytransform
  test_displaytransform.cpp
  ${CMAKE_SOURCE_DIR}/src/displaytransform.cpp
)

target_include_directories(test_displaytransform PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_displaytransform PRIVATE GTest::gtest_main Qt6::Gui)

gtest_discover_tests(test_displaytransform)

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

# Test executable for the archived-run store: the filesystem side of run
# history, which RunArchive's own tests do not touch.
add_executable(test_runhistory
  test_runhistory.cpp
  ${CMAKE_SOURCE_DIR}/src/runhistory.cpp
  ${CMAKE_SOURCE_DIR}/src/runarchive.cpp
  ${CMAKE_SOURCE_DIR}/src/runcompare.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdata.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/theme.cpp
)

target_include_directories(test_runhistory PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_runhistory PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_runhistory PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Svg Qt6::PrintSupport Qt6::Test)

gtest_discover_tests(test_runhistory)

# Test executable for the dump-image colour-map table, which the command
# builder and the settings swatches both read.
add_executable(test_colormaps
  test_colormaps.cpp
  ${CMAKE_SOURCE_DIR}/src/colormaps.cpp
)

target_include_directories(test_colormaps PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_colormaps PRIVATE GTest::gtest_main Qt6::Gui)

gtest_discover_tests(test_colormaps)

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

  # Guarded on SPARTA_GUI_HAVE_VTK rather than VTK_FOUND: the filter tests above
  # only need VTK to exist, but this one links the scene, and the scene is only
  # part of PROJECT_SOURCES when the application was configured to build it.
  if (SPARTA_GUI_HAVE_VTK)

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
    GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
    lepton_mini ads::qtadvanceddocking-qt6 ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_vtkscene MODULES ${VTK_LIBRARIES})
  if (SPARTA_GUI_USE_PLUGIN)
    target_compile_definitions(test_vtkscene PRIVATE SPARTA_GUI_USE_PLUGIN)
    target_include_directories(test_vtkscene PRIVATE ${SPARTA_PLUGINLIB_DIR})
  endif()
  # Discovered individually, so a failure names the case that failed. These
  # cover the bookkeeping and the file readers; the cases that actually render
  # skip without a display and are covered by the Xvfb run below.
  gtest_discover_tests(test_vtkscene)

  # VTK's off-screen rendering is off-screen only in the sense that no window
  # appears: on Linux it is still vtkXOpenGLRenderWindow speaking GLX, so it
  # needs an X server. gtest_discover_tests cannot wrap the per-case commands
  # in one (TEST_LAUNCHER is CMake 3.29 and this project targets 3.20), so the
  # rendering cases run as a single test under Xvfb. Without this the 3D
  # viewer has no coverage of what it puts on screen at all -- which is the
  # only thing a 3D viewer is for.
  # -a rather than a pinned display number: ctest runs these side by side with
  # the other Xvfb-wrapped suite, and two runs that each insist on one display
  # collide -- every case passes and xvfb-run still exits non-zero, so the suite
  # is reported as failed with nothing in its output to say why.
  add_test(NAME VtkScene.Rendering
    COMMAND ${XVFB_RUNNER} -a -s "-screen 0 1024x768x24" -w 1
    $<TARGET_FILE:test_vtkscene>
    --gtest_filter=VtkSceneRender.*:VtkSceneFiles.*)
  set_tests_properties(VtkScene.Rendering PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300 RESOURCE_LOCK xvfb)
  endif()
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

# Test executable for the SPARTA syntax highlighter. Needs Widgets because it
# builds real QTextDocuments, and helpers.cpp because the highlighter asks it
# which way round the application theme is.
add_executable(test_highlighter
  test_highlighter.cpp
  ${CMAKE_SOURCE_DIR}/src/highlighter.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/theme.cpp
)

target_include_directories(test_highlighter PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_highlighter PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_highlighter PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Svg)

gtest_discover_tests(test_highlighter)

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
# The editor tests quit the application, which it will not do while a modal is
# up -- and with no library to load, the first thing it puts up is a modal
# saying so.
list(APPEND PYTHON_TEST_ENVIRONMENT "SPARTA_PLUGIN_LIB=${_spa_test_lib}")

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
  # SPARTA_PLUGIN_LIB is the same library the live-window walks use. Without
  # it these suites drove an application that had no simulator behind it:
  # SPARTA_LIB_DIR pointed at the GUI's own build directory, which never
  # contains a libsparta, so nothing could be run and every check that needed
  # a completed run reported a missing control instead of a missing library.
  set(GUI_DRIVE_ENVIRONMENT
    "SPARTA_GUI=$<TARGET_FILE:sparta-gui>"
    "SPARTA_LIB_DIR=${CMAKE_BINARY_DIR}"
    "SPARTA_PLUGIN_LIB=${_spa_test_lib}"
    "SPARTA_FIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    # Absolute, and pinned to this checkout. The application resolves a relative
    # deck against its own working directory, not the harness's, so a relative
    # path opened nothing and left a "cannot open file" modal over every
    # screenshot that followed -- which then compared equal to the reference and
    # reported working actions as doing nothing.
    "SPARTA_EXAMPLES=${CMAKE_SOURCE_DIR}/../../examples"
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

  # Every menu action, judged against an idle reference: an action meant to
  # open something that leaves the screen byte-identical did nothing. This
  # suite has existed since the menus were built and was never registered, so
  # it had never run once.
  add_test(NAME GuiDrive.Actions
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_actions.py)
  set_tests_properties(GuiDrive.Actions PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/action-shots"
    TIMEOUT 900)

  # Activates every control of the running application in turn. The in-process
  # walker builds the dialogs directly and so cannot reach the main window's
  # menus, toolbar or dock chrome; this drives those, and it is what found the
  # paste crash. It takes a while because it drives a hundred controls and
  # photographs the screen after each one.
  add_test(NAME GuiDrive.Walker
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_walker.py)
  set_tests_properties(GuiDrive.Walker PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/walker-shots"
    TIMEOUT 1800)
else()
  message(STATUS "Skipping the visual and feature GUI suites: "
    "they need xdotool, a window manager (e.g. openbox) and a screenshooter")
endif()
endif()
