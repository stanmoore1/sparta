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

# Discard an existing environment whose interpreter cannot import tkinter.
#
# The probe above picks a Python that has it, but that only decides what a
# *new* environment is built from.  An environment left over from an earlier
# configure -- or restored from a snapshot, which is how this was hit -- keeps
# whatever bin/python pointed at then, and the build below has no reason to
# touch it.  The result is three PyAutoGUI tests failing on "You must install
# tkinter", with a pyvenv.cfg naming an interpreter that does have it: the
# config records what created the environment, not what bin/python resolves to
# now.  Ask the environment itself rather than believing the file.
if (EXISTS "${TESTENV_BINARY_DIR}/python")
  execute_process(COMMAND "${TESTENV_BINARY_DIR}/python" -c "import tkinter"
    RESULT_VARIABLE _testenv_tk OUTPUT_QUIET ERROR_QUIET)
  if (NOT _testenv_tk EQUAL 0)
    message(STATUS "Existing test environment cannot import tkinter: rebuilding it")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/testenv")
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/testenv.stamp")
  endif()
endif()

# A stamp file rather than the bin/ directory as the output: a directory's
# timestamp changes for reasons unrelated to whether the environment inside it
# is usable, so the recipe was considered up to date and skipped exactly when
# it most needed to run.
set(TESTENV_STAMP "${CMAKE_CURRENT_BINARY_DIR}/testenv.stamp")
add_custom_command(
  OUTPUT ${TESTENV_STAMP}
  # --clear so a directory that is already there is emptied first; without it
  # venv keeps the old bin/python symlinks and the interpreter probed above is
  # silently not the one the tests end up running
  COMMAND ${TESTENV_PYTHON} -m venv --clear "${CMAKE_CURRENT_BINARY_DIR}/testenv"
  COMMAND ${TESTENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install --upgrade pip
  COMMAND ${TESTENV_BINARY_DIR}/pip $ENV{PIP_OPTIONS} install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt --upgrade
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shooter.${SHOOTER} ${TESTENV_BINARY_DIR}/shooter
  COMMAND ${CMAKE_COMMAND} -E touch ${TESTENV_STAMP}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  # rebuilt when the requirements change, and when the interpreter it is built
  # from is replaced
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt ${TESTENV_PYTHON}
)
add_custom_target(setup_tests ALL
  ${CMAKE_COMMAND} -E echo "Setting up virtual environment for testing"
  DEPENDS ${TESTENV_STAMP}
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

# The standalone executable, for the one suite that needs a second opinion:
# the end-to-end workflow compares the numbers the application produced against
# the same deck run outside it. Without it that suite has nothing to check the
# application against and says so rather than reporting a pass.
set(SPARTA_TEST_BINARY "" CACHE FILEPATH
  "Standalone SPARTA used as the workflow suite's reference (empty: probe)")
if (SPARTA_TEST_BINARY)
  set(_spa_test_bin "${SPARTA_TEST_BINARY}")
else()
  get_filename_component(_spa_lib_dir "${_spa_test_lib}" DIRECTORY)
  file(GLOB _spa_bin_candidates
    "${_spa_lib_dir}/spa_*"
    "${CMAKE_SOURCE_DIR}/../../build*/src/spa_*"
    "${CMAKE_SOURCE_DIR}/../../src/spa_*")
  set(_spa_test_bin "")
  foreach(_cand IN LISTS _spa_bin_candidates)
    # the glob also matches libsparta's own name in some layouts
    if (NOT IS_DIRECTORY "${_cand}" AND NOT _cand MATCHES "\\.(so|a|dylib|dll)")
      set(_spa_test_bin "${_cand}")
      break()
    endif()
  endforeach()
endif()
if (_spa_test_bin AND EXISTS "${_spa_test_bin}")
  message(STATUS "Workflow reference run will use ${_spa_test_bin}")
else()
  set(_spa_test_bin "")
endif()

# The library interface itself, dlopened exactly as the plugin loader does it.
# This is the harness that proved the extended library API before there was any
# Qt code, and it had never been registered -- so the contract the whole GUI
# rests on was only ever checked by hand. It needs no display and no Qt; it
# needs a shared libsparta and the circle data files, which are in fixtures/.
if (_spa_test_lib)
  add_executable(test_library ${CMAKE_CURRENT_SOURCE_DIR}/test_library.cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(test_library PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

  # The deck reads data.circle/air.species/air.vss by bare name and the render
  # step writes frames next to them, so it gets a copy of the fixtures in a
  # directory of its own rather than writing into the source tree.
  set(_libtest_dir "${CMAKE_CURRENT_BINARY_DIR}/libtest")
  file(MAKE_DIRECTORY "${_libtest_dir}")
  foreach(_f data.circle air.species air.vss)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${_f}"
                   "${_libtest_dir}/${_f}" COPYONLY)
  endforeach()

  add_test(NAME Library.Interface
    COMMAND $<TARGET_FILE:test_library> "${_spa_test_lib}" "${_libtest_dir}")
  set_tests_properties(Library.Interface PROPERTIES TIMEOUT 900)
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
  # REPEAT UNTIL_PASS:3 on the three suites that need a display.
  #
  # About one full-suite run in five, one of them dies partway through with
  # "vtkXOpenGLRenderWindow: bad X server connection" -- the server goes away
  # mid-run, after several cases have already passed on it, so it is not a
  # startup collision: pinning a display number per suite changed nothing.
  # Each passes alone, and it predates the suites added here. The retry is a
  # backstop, not a diagnosis; the underlying fault is in how the VTK render
  # windows close their X connection and is worth its own look.
  #
  # RESOURCE_LOCK, because -a still races: two concurrent xvfb-run invocations
  # can both decide the same display is free before either starts a server on
  # it. The loser's cases all pass and xvfb-run exits non-zero anyway, so the
  # suite is reported as failed with nothing in its output to say why.
  set_tests_properties(MainWindow.All PROPERTIES TIMEOUT 300 RESOURCE_LOCK xvfb
  REPEAT UNTIL_PASS:3)
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
  SPARTA_DOC_DIR="${CMAKE_SOURCE_DIR}/../../doc/src"
  SPARTA_GUI_DOC_DIR="${CMAKE_SOURCE_DIR}/doc")
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

# The chart window: thermo output turned into plots, and the standalone plot
# window the same class becomes when handed a data file instead of a running
# simulation. It was the largest wholly untested source in the GUI. Handed a
# null SpartaGui it needs no simulator, but chartviewer.cpp includes
# spartagui.h, so this links the application the way test_mainwindow does.
add_executable(test_chartviewer
  test_chartviewer.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_chartviewer PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_chartviewer PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_chartviewer PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_chartviewer PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_chartviewer PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_chartviewer MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_chartviewer PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_chartviewer PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_chartviewer)

# The read-only file viewer, the shell window the two viewers share, and the
# band slider. The file viewer decompresses through six external programs and
# reports a file it cannot open in the window itself, which is what a user sees
# when a run wrote nothing. ViewerWindow builds an ImageViewer or a SlideShow,
# so this links the application.
add_executable(test_fileviewer
  test_fileviewer.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_fileviewer PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_fileviewer PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_fileviewer PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_fileviewer PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_fileviewer PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_fileviewer MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_fileviewer PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_fileviewer PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_fileviewer)

# The input editor. Its text transforms -- reformatting a command into aligned
# columns, commenting and uncommenting, and the completion lists built by
# parsing the buffer -- are what silently damage a deck if they are wrong, and
# were reachable only through the live walker, which types into the editor but
# never checks what came out.
add_executable(test_codeeditor
  test_codeeditor.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_codeeditor PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_codeeditor PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_codeeditor PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_codeeditor PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_codeeditor PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_codeeditor MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_codeeditor PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_codeeditor PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_codeeditor)

# The chart window's four modal dialogs: chart style, postprocess analysis, the
# Birch-Murnaghan column setup and its result, and the reference lines. Each was
# built on the stack inside a ChartWindow method and read a live ChartViewer as
# it went, so none could be constructed without a chart with data in it. Split
# out they are pure functions of plain structs, and this target links three leaf
# sources -- no chart, no simulator, no display.
add_executable(test_chartdialogs
  test_chartdialogs.cpp
  ${CMAKE_SOURCE_DIR}/src/chartdialogs.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
)
target_include_directories(test_chartdialogs PRIVATE ${CMAKE_SOURCE_DIR}/src)
# GTest::gtest, not gtest_main: the test brings its own main() so it can put Qt
# on the offscreen platform before the QApplication exists.
target_link_libraries(test_chartdialogs PRIVATE GTest::gtest Qt6::Widgets Qt6::Test)
gtest_discover_tests(test_chartdialogs)

# The slide show. Its sequence bookkeeping -- the active range, where "next"
# goes at the end of it, what Start and Stop do as images arrive -- needs no
# simulator and had no test: the live suites drive the buttons and photograph
# the result, which catches a crash but not an off-by-one in which image shows.
add_executable(test_slideshow
  test_slideshow.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_slideshow PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_slideshow PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_slideshow PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_slideshow PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_slideshow PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_slideshow MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_slideshow PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_slideshow PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_slideshow)

# What ChartWindow::postProcess() does after its dialog is answered: seven
# analyses that read the chart's data and put the answer back on it. Splitting
# the dialog out left this half still needing a chart with data in it -- which a
# ChartWindow loaded from a PlotData is, with no simulator involved. The test
# drives the modal dialogs from a timer and records what they reported.
add_executable(test_chartanalysis
  test_chartanalysis.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_chartanalysis PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_chartanalysis PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_chartanalysis PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_chartanalysis PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_chartanalysis PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_chartanalysis MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_chartanalysis PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_chartanalysis PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_chartanalysis)

# The rest of the chart window: the four ways data leaves it (PNG, DAT, CSV,
# YAML), the two dialogs that write back into it (style, reference lines) and
# the keys it claims for itself. All of it sat behind a QFileDialog or a modal,
# which is why it had never run. The exports matter more than their line count:
# chartsToPlotData() underneath them reads every column while the window shows
# one, so an export that dropped the off-screen columns would look right on
# screen and wrong in the file.
add_executable(test_chartexport
  test_chartexport.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_chartexport PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_chartexport PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_chartexport PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_chartexport PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_chartexport PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_chartexport MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_chartexport PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_chartexport PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_chartexport)

# The two main-window dialogs that write back into the running application:
# Preferences and Set Variables, plus the Open Example menu they can rebuild.
# Both are on test_mainwindow.cpp's do-not-trigger list because each opens a
# modal nobody was there to answer, so what happens after the dialog is accepted
# -- which is the whole point of both -- had never run.
add_executable(test_mainwindowprefs
  test_mainwindowprefs.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_mainwindowprefs PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_mainwindowprefs PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_mainwindowprefs PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_mainwindowprefs PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_mainwindowprefs PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_mainwindowprefs MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_mainwindowprefs PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_mainwindowprefs PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
# the window loads the SPARTA library into this process
add_test(NAME MainWindow.Prefs COMMAND $<TARGET_FILE:test_mainwindowprefs>)
set_tests_properties(MainWindow.Prefs PROPERTIES TIMEOUT 900 RESOURCE_LOCK "sparta_run")

# The File menu, and the workers behind it. test_mainwindow.cpp keeps a list of
# actions it must not trigger because each opens a modal nobody can answer, and
# nearly all of them are the file actions -- which is why opening, saving,
# viewing and inspecting a file were the largest uncovered block in the
# application. This drives the QFileDialog from a timer (AA_DontUseNativeDialogs
# makes it an ordinary widget) and calls the workers directly.
add_executable(test_mainwindowfiles
  test_mainwindowfiles.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_mainwindowfiles PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_mainwindowfiles PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_mainwindowfiles PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_mainwindowfiles PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_mainwindowfiles PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_mainwindowfiles MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_mainwindowfiles PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_mainwindowfiles PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_mainwindowfiles)

# Crash recovery: the autosave copy of an unsaved buffer and the offer to
# restore it.  Its failure mode is invisible until the session it was meant to
# survive, and it writes files on a timer next to the user's own deck, so both
# directions need checking.  Driven through the real triggers -- the autosave
# timer, the constructor's offer, and the clear on save.
add_executable(test_recovery
  test_recovery.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_recovery PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_recovery PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_recovery PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_recovery PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_recovery PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_recovery MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_recovery PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_recovery PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_recovery)

# Plotting a data file: SpartaGui::plotDataFile() and ChartWindow::addDataFile().
# The parsers and the column picker are covered separately; neither end of the
# path that joins them had been driven, so nothing checked that the columns the
# user picked are the ones that reach the chart.  A curve from the wrong column
# is still a curve, and it is labelled from the same picker.
add_executable(test_datafileplot
  test_datafileplot.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_datafileplot PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_datafileplot PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_datafileplot PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_datafileplot PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_datafileplot PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_datafileplot MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_datafileplot PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_datafileplot PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_datafileplot)

# Steering the snapshot with the mouse: ImageViewer::eventFilter().  Drag to
# rotate, shift-drag to pan, wheel to zoom -- how the viewer is actually used,
# and never run under test.  The view state is private, so the checks read it
# back out of the dump image command the viewer emits, which is the same state
# the render uses.
add_executable(test_imageviewerinput
  test_imageviewerinput.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_imageviewerinput PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_imageviewerinput PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_imageviewerinput PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_imageviewerinput PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_imageviewerinput PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_imageviewerinput MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_imageviewerinput PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_imageviewerinput PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# each case loads the SPARTA library into this process and chdirs to build a
# scene, so they must not interleave with the other live suites
add_test(NAME ImageViewer.Input COMMAND $<TARGET_FILE:test_imageviewerinput>)
set_tests_properties(ImageViewer.Input PROPERTIES TIMEOUT 900 RESOURCE_LOCK "sparta_run")

# The snapshot viewer's toolbar, its colour files, and the keys it claims.
# Every toggle slot reads sender(), so none of them can be invoked directly --
# they do nothing unless a real button emitted the click, which is why they had
# never run. What each click did is read back out of the dump image command the
# viewer puts on the clipboard, so a toggle that flipped the button but not the
# render setting is told apart from one that did both.
add_executable(test_imageviewerbuttons
  test_imageviewerbuttons.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_imageviewerbuttons PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_imageviewerbuttons PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_imageviewerbuttons PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_imageviewerbuttons PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_imageviewerbuttons PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_imageviewerbuttons MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_imageviewerbuttons PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_imageviewerbuttons PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
add_test(NAME ImageViewer.Buttons COMMAND $<TARGET_FILE:test_imageviewerbuttons>)
set_tests_properties(ImageViewer.Buttons PROPERTIES TIMEOUT 900 RESOURCE_LOCK "sparta_run")

# Inspecting a restart file.  The one place the application opens somebody's
# saved simulation without running it -- and not a read-only path: it loads the
# file into the live instance, replacing what was there, and writes a temporary
# log beside the user's file.
add_executable(test_inspect
  test_inspect.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_inspect PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_inspect PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_inspect PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_inspect PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_inspect PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_inspect MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_inspect PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_inspect PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# it runs a deck to produce the restart file, so it takes the run lock; under
# Xvfb because inspecting opens the snapshot viewer, which renders
add_test(NAME Inspect.All COMMAND ${XVFB_RUNNER} -a -s "-screen 0 1024x768x24" -w 1
  $<TARGET_FILE:test_inspect>)
set_tests_properties(Inspect.All PROPERTIES TIMEOUT 900 RESOURCE_LOCK "xvfb;sparta_run")

# Integrity checking of downloaded files.  This is the only code that decides
# whether a file fetched off the network may be kept, and none of it had ever
# run: a parser that misses the entry, or a comparison that always agrees,
# downgrades the check to nothing while still reporting success.  Driven over
# file:// URLs, so the real fetch-parse-compare path runs with no network.
add_executable(test_checksum
  test_checksum.cpp
  ${CMAKE_SOURCE_DIR}/src/urldownloader.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
)
target_include_directories(test_checksum PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_checksum PRIVATE GTest::gtest Qt6::Widgets Qt6::Network Qt6::Svg)

gtest_discover_tests(test_checksum)

# The run path: doRun(), archiveFinishedRun(), continueRestart() and
# renderVtkSnapshot(). Three of the four need a simulation actually running,
# which this suite can have -- the same shared libsparta the window loads
# anyway, driven through the window's own Run action on a deck small enough to
# finish in a moment. One ctest entry rather than per-case discovery: every case
# loads the SPARTA library into this process, runs a simulation and chdirs to do
# it, so the cases must not race each other or the other live-window suites.
add_executable(test_run
  test_run.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_run PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_run PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_run PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_run PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_run PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_run MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_run PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_run PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
# Under Xvfb: VTK renders through its own X connection even when Qt is on the
# offscreen platform, so the 3D snapshot cases abort on "bad X server
# connection" without a display. Both locks -- the Xvfb one it shares with the
# other display-using suites, and its own, because each case loads the SPARTA
# library into this process and chdirs to run a simulation.
add_test(NAME Run.All COMMAND ${XVFB_RUNNER} -a -s "-screen 0 1024x768x24" -w 1
  $<TARGET_FILE:test_run>)
set_tests_properties(Run.All PROPERTIES TIMEOUT 900 RESOURCE_LOCK "xvfb;sparta_run"
  REPEAT UNTIL_PASS:3)

# The surface import wizard: the six pages on top of the parsers and command
# builders that test_stlimport.cpp already covers. It needs no simulator -- it
# checks its SpartaWrapper before every use, and with none it falls back to the
# preflight watertightness heuristic and its own mesh renderer.
add_executable(test_stlimportwizard
  test_stlimportwizard.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_stlimportwizard PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_stlimportwizard PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_stlimportwizard PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_stlimportwizard PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_stlimportwizard PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_stlimportwizard MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_stlimportwizard PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_stlimportwizard PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_stlimportwizard)

# The wizard's SPARTA-facing half, against a live simulator: the domain it
# builds for the surface it just imported, which is where every surface-based
# simulation starts. Its failures are quiet -- a box that does not enclose the
# geometry still runs -- so the assertions are against the box SPARTA ended up
# with, read back through the library.
add_executable(test_stlwizardlive
  test_stlwizardlive.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_stlwizardlive PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_stlwizardlive PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_stlwizardlive PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_stlwizardlive PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_stlwizardlive PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_stlwizardlive MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_stlwizardlive PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_stlwizardlive PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# One process, one SPARTA instance: each case clears and rebuilds it, so they
# must not interleave with the other suites that load the library.
add_test(NAME StlWizard.Live COMMAND $<TARGET_FILE:test_stlwizardlive>)
set_tests_properties(StlWizard.Live PROPERTIES TIMEOUT 900 RESOURCE_LOCK "sparta_run")

# The species colour file: what "Save Colors" writes and "Load Colors" reads.
# It is the one thing the image viewer persists that a user can hand to someone
# else, so it carries a header saying what it is and a revision to refuse a
# format it does not understand. Both ends go through a file dialog, which is
# why neither end had been checked.
add_executable(test_jsoncolors
  test_jsoncolors.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_jsoncolors PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_jsoncolors PRIVATE SPARTA_GUI_VERSION="${PROJECT_VERSION}")
target_link_libraries(test_jsoncolors PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_jsoncolors PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_jsoncolors PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_jsoncolors MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_jsoncolors PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_jsoncolors PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
gtest_discover_tests(test_jsoncolors)

# The parametric sweep driver and the library boundary it reads through.
# SweepController runs unattended over many combinations and hands back a table
# the user reads as science; its failures are quiet, so every assertion is
# against an arithmetic answer. Live, like Run.All, and under the same locks:
# each case loads the SPARTA library into this process and chdirs to run.
add_executable(test_sweeprun
  test_sweeprun.cpp
  ${GUI_WALKER_SOURCES}
)
target_include_directories(test_sweeprun PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider
)
target_compile_definitions(test_sweeprun PRIVATE
  SPARTA_GUI_VERSION="${PROJECT_VERSION}"
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_sweeprun PRIVATE
  GTest::gtest Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Svg Qt6::Test
  lepton_mini ads::qtadvanceddocking-qt6)
if (SPARTA_GUI_HAVE_VTK)
  target_compile_definitions(test_sweeprun PRIVATE SPARTA_GUI_HAVE_VTK)
  target_link_libraries(test_sweeprun PRIVATE ${VTK_LIBRARIES})
  vtk_module_autoinit(TARGETS test_sweeprun MODULES ${VTK_LIBRARIES})
endif()
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_sweeprun PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_sweeprun PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()
add_test(NAME Sweep.All COMMAND ${XVFB_RUNNER} -a -s "-screen 0 1024x768x24" -w 1
  $<TARGET_FILE:test_sweeprun>)
set_tests_properties(Sweep.All PROPERTIES TIMEOUT 1800 RESOURCE_LOCK "xvfb;sparta_run"
  REPEAT UNTIL_PASS:3)

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

# The export dialog's conversion run.  The pure part is covered above; this is
# the orchestration around it -- locating the script, refusing to start without
# the tools, replacing stale output, running pvpython in the right directory and
# deciding afterwards whether it worked.  ParaView is not needed: the tests
# supply a stub pvpython that records how it was called and exits how they ask.
add_executable(test_paraviewdialog
  test_paraviewdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/paraviewdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/paraviewexport.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
)
target_include_directories(test_paraviewdialog PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_paraviewdialog PRIVATE
  GTest::gtest Qt6::Widgets Qt6::Svg)

gtest_discover_tests(test_paraviewdialog)

# Test executable for the parametric-sweep expansion/reducers
add_executable(test_sweepspec
  test_sweepspec.cpp
  ${CMAKE_SOURCE_DIR}/src/sweepspec.cpp
)

target_include_directories(test_sweepspec PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sweepspec PRIVATE GTest::gtest_main Qt6::Widgets)

gtest_discover_tests(test_sweepspec)

# The menu walker feeding the command palette, the generated shortcut sheet and
# the status-tip pass.  Own main(): QMenu needs a QApplication.
add_executable(test_actionscan
  test_actionscan.cpp
  ${CMAKE_SOURCE_DIR}/src/actionscan.cpp
)

target_include_directories(test_actionscan PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_actionscan PRIVATE GTest::gtest Qt6::Widgets)

gtest_discover_tests(test_actionscan)

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

# The per-surface extraction path against a live simulation.  extractCompute and
# extractFix are the only wrapper calls that return a pointer into SPARTA's own
# memory, and the surface report is their sole consumer; the reduction core
# above has only ever seen hand-written arrays.  A wrong style constant or a
# stride mismatch would yield a report full of plausible numbers.
add_executable(test_surfreportlive
  test_surfreportlive.cpp
  ${CMAKE_SOURCE_DIR}/src/surfreportdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/surfreport.cpp
  ${CMAKE_SOURCE_DIR}/src/casemodel.cpp
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${PLUGIN_LOADER_SRC}
)
target_include_directories(test_surfreportlive PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_surfreportlive PRIVATE
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}"
  SPARTA_SURF_FIXTURES="${CMAKE_SOURCE_DIR}/../../examples/circle")
target_link_libraries(test_surfreportlive PRIVATE GTest::gtest_main Qt6::Widgets)
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_surfreportlive PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_surfreportlive PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# one process, one SPARTA instance shared by the suite: the run is the expensive
# part and every case reads the same finished state
add_test(NAME SurfReport.Live COMMAND $<TARGET_FILE:test_surfreportlive>)
set_tests_properties(SurfReport.Live PROPERTIES TIMEOUT 600 RESOURCE_LOCK "sparta_run")

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
    --gtest_filter=VtkSceneRender.*:VtkSceneFiles.*:SceneProbe.*)
  set_tests_properties(VtkScene.Rendering PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1" TIMEOUT 300 RESOURCE_LOCK xvfb REPEAT UNTIL_PASS:3)
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
# Qt6::Svg for the dialog's icons, and the resource bundle they live in
target_sources(test_movieimport PRIVATE ${TEST_RESOURCE_SOURCES})
target_link_libraries(test_movieimport PRIVATE GTest::gtest Qt6::Widgets Qt6::Svg)

gtest_discover_tests(test_movieimport)

# The same code against a real movie.  The suite above parses ffprobe output
# from hand-written strings; this runs ffmpeg both ways -- building a movie
# whose frames are individually identifiable by colour, then checking that the
# frames extracted are the ones selected.  A stride that is off by one produces
# a slide show that looks entirely normal and is the wrong part of the run.
add_executable(test_movielive
  test_movielive.cpp
  ${CMAKE_SOURCE_DIR}/src/movieimport.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
)
target_include_directories(test_movielive PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_movielive PRIVATE GTest::gtest Qt6::Widgets Qt6::Svg)

# one ctest entry: each case encodes and decodes a movie, and running several
# ffmpeg jobs beside the rest of the suite is what makes them time out
add_test(NAME Movie.Live COMMAND $<TARGET_FILE:test_movielive>)
set_tests_properties(Movie.Live PROPERTIES TIMEOUT 900 RESOURCE_LOCK "ffmpeg")

# Loading the SPARTA library, and what the wrapper does when that fails.  Every
# other suite uses a library that loads; this one uses the ones that do not.  In
# plugin mode every call goes through a function table that is absent from
# application start until a library is chosen, so a call with no guard jumps
# through it -- writing these found two.
add_executable(test_wrapperload
  test_wrapperload.cpp
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
  ${PLUGIN_LOADER_SRC}
)
target_include_directories(test_wrapperload PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_wrapperload PRIVATE
  SPARTA_TEST_LIBRARY_PATH="${_spa_test_lib}")
target_link_libraries(test_wrapperload PRIVATE GTest::gtest Qt6::Widgets Qt6::Svg)
if (SPARTA_GUI_USE_PLUGIN)
  target_compile_definitions(test_wrapperload PRIVATE SPARTA_GUI_USE_PLUGIN)
  target_include_directories(test_wrapperload PRIVATE ${SPARTA_PLUGINLIB_DIR})
endif()

# it opens and closes SPARTA instances of its own
add_test(NAME Wrapper.Load COMMAND $<TARGET_FILE:test_wrapperload>)
set_tests_properties(Wrapper.Load PROPERTIES TIMEOUT 300 RESOURCE_LOCK "sparta_run")

# Saving a snapshot to a file: the only way a rendered frame leaves the
# application, and never run.  Qt writes a few formats directly; anything else
# goes through a temporary PNG and ImageMagick, which is the half that had no
# coverage at all.
add_executable(test_exportimage
  test_exportimage.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
)
target_include_directories(test_exportimage PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_exportimage PRIVATE GTest::gtest Qt6::Widgets Qt6::Svg)

gtest_discover_tests(test_exportimage)

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
                     PROPERTIES ENVIRONMENT "${PYTHON_TEST_ENVIRONMENT}"
                                RESOURCE_LOCK "xvfb")

# test basic SPARTA-GUI editor tasks
add_test(NAME Framebuffer.GUIEditorChecks
  COMMAND ${XVFB_RUNNER} -n 13 -s "-screen 0 1024x768x24" -w 1
  ${TESTENV_BINARY_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_edit.py
)
# The same "xvfb" lock the GuiDrive suites below take.  Without it this was the
# one display-driven test ctest would schedule alongside them: it drives the
# pointer and keyboard through PyAutoGUI and waits fixed intervals for windows
# to settle, so four GUI applications competing for the CPU turned every one of
# those waits into a race.  Under -j the whole set failed together and each one
# passed alone, which reads like eight broken tests rather than one missing
# property.
set_tests_properties(Framebuffer.GUIEditorChecks
                     PROPERTIES ENVIRONMENT "${PYTHON_TEST_ENVIRONMENT}"
                                RESOURCE_LOCK "xvfb")

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

  # All four hold the xvfb lock. They use distinct display numbers, so they do
  # not collide over a server -- but each one runs an X server, a window
  # manager, an accessibility bus and the application, then compares
  # screenshots against references. Four of those at once is a timing race by
  # construction: a dialog photographed before it has painted reads as an
  # action that did nothing.
  add_test(NAME GuiDrive.VisualEffects
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_visual.py)
  set_tests_properties(GuiDrive.VisualEffects PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/visual-shots"
    TIMEOUT 600 RESOURCE_LOCK xvfb)

  add_test(NAME GuiDrive.Features
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_features.py)
  set_tests_properties(GuiDrive.Features PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/feature-shots"
    TIMEOUT 900 RESOURCE_LOCK xvfb)

  # Every menu action, judged against an idle reference: an action meant to
  # open something that leaves the screen byte-identical did nothing. This
  # suite has existed since the menus were built and was never registered, so
  # it had never run once.
  add_test(NAME GuiDrive.Actions
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_actions.py)
  set_tests_properties(GuiDrive.Actions PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/action-shots"
    TIMEOUT 900 RESOURCE_LOCK xvfb)

  # Activates every control of the running application in turn. The in-process
  # walker builds the dialogs directly and so cannot reach the main window's
  # menus, toolbar or dock chrome; this drives those, and it is what found the
  # paste crash. It takes a while because it drives a hundred controls and
  # photographs the screen after each one.
  add_test(NAME GuiDrive.Walker
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gui_walker.py)
  set_tests_properties(GuiDrive.Walker PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/walker-shots"
    TIMEOUT 1800 RESOURCE_LOCK xvfb)

  # What the application does before it has a main window: src/main.cpp. main()
  # is one function ending in app.exec(), so nothing in it is reachable from an
  # in-process test -- which is why it was the least covered file in the
  # project. This runs the real binary as a subprocess for each command line
  # and checks the exit status and the window it left behind.
  add_test(NAME GuiDrive.Startup
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_startup.py)
  set_tests_properties(GuiDrive.Startup PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT}"
    TIMEOUT 900 RESOURCE_LOCK xvfb)

  # The missing-library dialog: the first thing a user without a SPARTA shared
  # library sees, and the one place the application can strand them. Every
  # branch of it ends in exit(1) or in a re-exec, so it cannot be driven in
  # process -- an in-process test would take the runner with it. The buttons
  # are pressed through accessibility, which is also the only way to reach the
  # middle one: Enter hits the default (Download) and Escape hits Exit.
  add_test(NAME GuiDrive.PluginSetup
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pluginsetup.py)
  set_tests_properties(GuiDrive.PluginSetup PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/plugin-shots"
    TIMEOUT 900 RESOURCE_LOCK xvfb)

  # One DSMC study from beginning to end, the way a user does it: open the
  # example, check the input, run it, chart the result, look at the output,
  # render the geometry, read the surface report and the run history, export
  # for ParaView. Every other suite asks whether a control reacts; this one
  # asks whether the answer is right, by running the same deck outside the
  # application and requiring the two stats tables to agree column for column.
  # The physics is checked too -- the domain fills to a plateau, the surface
  # collision rate settles about a steady mean -- so that a run which agrees
  # with a broken reference still cannot pass.
  add_test(NAME GuiDrive.Workflow
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_workflow.py)
  set_tests_properties(GuiDrive.Workflow PROPERTIES
    ENVIRONMENT "${GUI_DRIVE_ENVIRONMENT};SHOT_DIR=${CMAKE_CURRENT_BINARY_DIR}/workflow-shots;SPARTA_BIN=${_spa_test_bin}"
    TIMEOUT 1800 RESOURCE_LOCK xvfb)
else()
  message(STATUS "Skipping the visual and feature GUI suites: "
    "they need xdotool, a window manager (e.g. openbox) and a screenshooter")
endif()
endif()
