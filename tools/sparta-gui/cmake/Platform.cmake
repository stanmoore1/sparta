##########################################################################
# Platform, compiler, and build-mode specific setup for the application.
# Runs before the sparta-gui target is created and provides the
# PLUGIN_LOADER_SRC, ICON_RC_FILE, and MACOSX_* variables consumed by
# the qt_add_executable() call and the packaging targets.
##########################################################################

# by default, install into $HOME/.local (not /usr/local),
# so that no root access (and sudo) is needed
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Default install path" FORCE)
endif()

# ugly hacks for MSVC which by default always reports an old C++ standard in
# the __cplusplus macro and prints lots of pointless warnings about "unsafe" functions
if(MSVC)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/Zc:__cplusplus)
    add_compile_options(/wd4244)
    add_compile_options(/wd4267)
    add_compile_options(/wd4250)
    add_compile_options(/EHsc)
    add_compile_options(/utf-8)
    add_compile_options(/nologo)
    add_link_options(/nologo)
    add_link_options(/release)
  endif()
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(SPARTA_PLUGINLIB_DIR ${CMAKE_SOURCE_DIR}/plugin)
if(SPARTA_GUI_USE_PLUGIN)
  enable_language(C)
  set(PLUGIN_LOADER_SRC ${SPARTA_PLUGINLIB_DIR}/libspartaplugin.c)
endif()

# include resource compiler to embed icons into the executable on Windows
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  enable_language(RC)
  set(ICON_RC_FILE ${CMAKE_SOURCE_DIR}/resources/spaicons.rc)
endif()

# icon, readme, and installer resources for the macOS app bundle; prefer
# the files from the SPARTA source tree when compiling with it
if(APPLE)
  if(SPARTA_SOURCE_DIR)
    set(MACOSX_ICON_FILE ${SPARTA_SOURCE_DIR}/../cmake/packaging/sparta-gui.icns)
    set(MACOSX_README_FILE ${SPARTA_SOURCE_DIR}/../cmake/packaging/README.macos)
    set(MACOSX_BACKGROUND_FILE ${SPARTA_SOURCE_DIR}/../cmake/packaging/SPARTA_DMG_Background.png)
    set(MACOSX_PLIST_FILE ${SPARTA_SOURCE_DIR}/../cmake/packaging/MacOSXBundleInfo.plist.in)
  else()
    set(MACOSX_ICON_FILE ${CMAKE_SOURCE_DIR}/packaging/sparta-gui.icns)
    set(MACOSX_README_FILE ${CMAKE_SOURCE_DIR}/packaging/README.macos)
    set(MACOSX_BACKGROUND_FILE ${CMAKE_SOURCE_DIR}/packaging/SPARTA_DMG_Background.png)
    set(MACOSX_PLIST_FILE ${CMAKE_SOURCE_DIR}/packaging/MacOSXBundleInfo.plist.in)
  endif()
endif()
