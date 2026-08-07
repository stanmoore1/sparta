##########################################################################
# All source files of the sparta-gui executable plus the bundled
# lepton_mini support library.  New .cpp/.h file pairs in src/ must be
# added to the PROJECT_SOURCES list below; Qt's AUTOMOC picks them up
# automatically once listed.  Expects Qt6 to be found and the variables
# from Platform.cmake to be set.
##########################################################################

set(PROJECT_SOURCES
  ${CMAKE_SOURCE_DIR}/src/main.cpp
  ${CMAKE_SOURCE_DIR}/src/spartagui.cpp
  ${CMAKE_SOURCE_DIR}/src/spartagui.h
  ${CMAKE_SOURCE_DIR}/src/aboutdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/aboutdialog.h
  ${CMAKE_SOURCE_DIR}/src/actionmetadata.cpp
  ${CMAKE_SOURCE_DIR}/src/actionmetadata.h
  ${CMAKE_SOURCE_DIR}/src/actionscan.cpp
  ${CMAKE_SOURCE_DIR}/src/actionscan.h
  ${CMAKE_SOURCE_DIR}/src/analysis.cpp
  ${CMAKE_SOURCE_DIR}/src/analysis.h
  ${CMAKE_SOURCE_DIR}/src/chartdialogs.cpp
  ${CMAKE_SOURCE_DIR}/src/chartviewer.cpp
  ${CMAKE_SOURCE_DIR}/src/chartdialogs.h
  ${CMAKE_SOURCE_DIR}/src/chartviewer.h
  ${CMAKE_SOURCE_DIR}/src/plotwidget.cpp
  ${CMAKE_SOURCE_DIR}/src/plotwidget.h
  ${CMAKE_SOURCE_DIR}/src/plotseries.h
  ${CMAKE_SOURCE_DIR}/src/plotaxismath.cpp
  ${CMAKE_SOURCE_DIR}/src/plotaxismath.h
  ${CMAKE_SOURCE_DIR}/src/codeeditor.cpp
  ${CMAKE_SOURCE_DIR}/src/codeeditor.h
  ${CMAKE_SOURCE_DIR}/src/colormaps.cpp
  ${CMAKE_SOURCE_DIR}/src/colormaps.h
  ${CMAKE_SOURCE_DIR}/src/constants.h
  ${CMAKE_SOURCE_DIR}/src/customfunc.cpp
  ${CMAKE_SOURCE_DIR}/src/customfunc.h
  ${CMAKE_SOURCE_DIR}/src/dockpanels.cpp
  ${CMAKE_SOURCE_DIR}/src/dockpanels.h
  ${CMAKE_SOURCE_DIR}/src/dumpimage.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimage.h
  ${CMAKE_SOURCE_DIR}/src/fileviewer.cpp
  ${CMAKE_SOURCE_DIR}/src/fileviewer.h
  ${CMAKE_SOURCE_DIR}/src/findandreplace.cpp
  ${CMAKE_SOURCE_DIR}/src/findandreplace.h
  ${CMAKE_SOURCE_DIR}/src/fitting.cpp
  ${CMAKE_SOURCE_DIR}/src/fitting.h
  ${CMAKE_SOURCE_DIR}/src/flagwarnings.cpp
  ${CMAKE_SOURCE_DIR}/src/flagwarnings.h
  ${CMAKE_SOURCE_DIR}/src/helpers.cpp
  ${CMAKE_SOURCE_DIR}/src/helpers.h
  ${CMAKE_SOURCE_DIR}/src/highlighter.cpp
  ${CMAKE_SOURCE_DIR}/src/highlighter.h
  ${CMAKE_SOURCE_DIR}/src/imagecache.cpp
  ${CMAKE_SOURCE_DIR}/src/imagecache.h
  ${CMAKE_SOURCE_DIR}/src/casemodel.cpp
  ${CMAKE_SOURCE_DIR}/src/casemodel.h
  ${CMAKE_SOURCE_DIR}/src/inputcheck.cpp
  ${CMAKE_SOURCE_DIR}/src/inputcheck.h
  ${CMAKE_SOURCE_DIR}/src/imageviewer.cpp
  ${CMAKE_SOURCE_DIR}/src/imageviewer.h
  ${CMAKE_SOURCE_DIR}/src/imageviewer_internal.h
  ${CMAKE_SOURCE_DIR}/src/imageviewersettings.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimagesettingsdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/dumpimagesettingsdialog.h
  ${CMAKE_SOURCE_DIR}/src/runcompare.cpp
  ${CMAKE_SOURCE_DIR}/src/runcompare.h
  ${CMAKE_SOURCE_DIR}/src/spartarunner.cpp
  ${CMAKE_SOURCE_DIR}/src/spartarunner.h
  ${CMAKE_SOURCE_DIR}/src/surfreport.cpp
  ${CMAKE_SOURCE_DIR}/src/surfreport.h
  ${CMAKE_SOURCE_DIR}/src/surfreportdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/surfreportdialog.h
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/spartawrapper.h
  ${CMAKE_SOURCE_DIR}/src/leastsquares.cpp
  ${CMAKE_SOURCE_DIR}/src/leastsquares.h
  ${CMAKE_SOURCE_DIR}/src/levmar.cpp
  ${CMAKE_SOURCE_DIR}/src/levmar.h
  ${CMAKE_SOURCE_DIR}/src/linenumberarea.h
  ${CMAKE_SOURCE_DIR}/src/logwindow.cpp
  ${CMAKE_SOURCE_DIR}/src/logwindow.h
  ${CMAKE_SOURCE_DIR}/src/movieimport.cpp
  ${CMAKE_SOURCE_DIR}/src/movieimport.h
  ${CMAKE_SOURCE_DIR}/src/plotdata.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdata.h
  ${CMAKE_SOURCE_DIR}/src/paraviewdialog.cpp
  ${CMAKE_SOURCE_DIR}/src/paraviewdialog.h
  ${CMAKE_SOURCE_DIR}/src/paraviewexport.cpp
  ${CMAKE_SOURCE_DIR}/src/paraviewexport.h
  ${CMAKE_SOURCE_DIR}/src/plotdatadialog.cpp
  ${CMAKE_SOURCE_DIR}/src/plotdatadialog.h
  ${CMAKE_SOURCE_DIR}/src/runarchive.cpp
  ${CMAKE_SOURCE_DIR}/src/runarchive.h
  ${CMAKE_SOURCE_DIR}/src/runhistory.cpp
  ${CMAKE_SOURCE_DIR}/src/runhistory.h
  ${CMAKE_SOURCE_DIR}/src/preferences.cpp
  ${CMAKE_SOURCE_DIR}/src/preferences.h
  ${CMAKE_SOURCE_DIR}/src/qaddon.cpp
  ${CMAKE_SOURCE_DIR}/src/qaddon.h
  ${CMAKE_SOURCE_DIR}/src/rangebandslider.cpp
  ${CMAKE_SOURCE_DIR}/src/rangebandslider.h
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider/rangeslider.cpp
  ${CMAKE_SOURCE_DIR}/thirdparty/rangeslider/rangeslider.h
  ${CMAKE_SOURCE_DIR}/src/setvariables.cpp
  ${CMAKE_SOURCE_DIR}/src/setvariables.h
  ${CMAKE_SOURCE_DIR}/src/slideshow.cpp
  ${CMAKE_SOURCE_DIR}/src/slideshow.h
  ${CMAKE_SOURCE_DIR}/src/displaytransform.cpp
  ${CMAKE_SOURCE_DIR}/src/displaytransform.h
  ${CMAKE_SOURCE_DIR}/src/viewerdisplay.cpp
  ${CMAKE_SOURCE_DIR}/src/viewerdisplay.h
  ${CMAKE_SOURCE_DIR}/src/viewerpanel.cpp
  ${CMAKE_SOURCE_DIR}/src/viewerpanel.h
  ${CMAKE_SOURCE_DIR}/src/viewersource.h
  ${CMAKE_SOURCE_DIR}/src/viewerwindow.cpp
  ${CMAKE_SOURCE_DIR}/src/viewerwindow.h
  ${CMAKE_SOURCE_DIR}/src/snippets.cpp
  ${CMAKE_SOURCE_DIR}/src/snippets.h
  ${CMAKE_SOURCE_DIR}/src/stdcapture.cpp
  ${CMAKE_SOURCE_DIR}/src/stdcapture.h
  ${CMAKE_SOURCE_DIR}/src/stlimport.cpp
  ${CMAKE_SOURCE_DIR}/src/stlimport.h
  ${CMAKE_SOURCE_DIR}/src/stlimportwizard.cpp
  ${CMAKE_SOURCE_DIR}/src/stlimportwizard.h
  ${CMAKE_SOURCE_DIR}/src/sweepspec.cpp
  ${CMAKE_SOURCE_DIR}/src/sweepspec.h
  ${CMAKE_SOURCE_DIR}/src/sweeppanel.cpp
  ${CMAKE_SOURCE_DIR}/src/sweeppanel.h
  ${CMAKE_SOURCE_DIR}/src/theme.cpp
  ${CMAKE_SOURCE_DIR}/src/theme.h
  ${CMAKE_SOURCE_DIR}/src/urldownloader.cpp
  ${CMAKE_SOURCE_DIR}/src/urldownloader.h
  ${CMAKE_SOURCE_DIR}/src/welcomescreen.cpp
  ${CMAKE_SOURCE_DIR}/src/welcomescreen.h
  ${PLUGIN_LOADER_SRC}
  ${ICON_RC_FILE}
)
qt6_add_resources(PROJECT_SOURCES resources/spartagui.qrc)

# vendored, JIT-less subset of the Lepton expression parser (namespace
# LeptonMini) used for custom-function plotting and nonlinear fits.
add_library(lepton_mini STATIC
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/src/ExpressionProgram.cpp
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/src/ExpressionTreeNode.cpp
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/src/Operation.cpp
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/src/ParsedExpression.cpp
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/src/Parser.cpp
)
target_include_directories(lepton_mini PUBLIC
  ${CMAKE_SOURCE_DIR}/thirdparty/lepton_mini/include)
# on Windows: build/consume as a static library (no dllexport/dllimport)
target_compile_definitions(lepton_mini
  PRIVATE LEPTON_BUILDING_STATIC_LIBRARY
  PUBLIC LEPTON_USE_STATIC_LIBRARIES)
set_target_properties(lepton_mini PROPERTIES POSITION_INDEPENDENT_CODE ON)
