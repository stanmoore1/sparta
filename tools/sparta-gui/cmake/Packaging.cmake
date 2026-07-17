##########################################################################
# Packaging targets: flatpak bundle, macOS app bundle and .dmg image,
# Windows deployment/NSIS installer, Linux .tar.gz bundle, and the
# source tarball.  The binary packaging targets are only available when
# compiling in plugin mode; when linking to SPARTA we are compiled as an
# external project and then SPARTA provides the packaging targets.
# Expects the sparta-gui target and the MACOSX_* variables from
# Platform.cmake.  Must be included from the top-level CMakeLists.txt
# (the install rules use CMAKE_CURRENT_SOURCE_DIR).
##########################################################################

if (SPARTA_GUI_USE_PLUGIN AND NOT BUILD_DOC_ONLY)
  # build SPARTA-GUI as flatpak
  find_program(FLATPAK_COMMAND flatpak DOC "Path to flatpak command")
  find_program(FLATPAK_BUILDER flatpak-builder DOC "Path to flatpak-builder command")
  if(FLATPAK_COMMAND AND FLATPAK_BUILDER)
    set(FLATPAK_BUNDLE "SPARTA-GUI-Linux-x86_64-${PROJECT_VERSION}.flatpak")
    add_custom_target(flatpak
      COMMAND ${FLATPAK_COMMAND} --user remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
      COMMAND ${FLATPAK_BUILDER} --force-clean --verbose --repo=${CMAKE_CURRENT_BINARY_DIR}/flatpak-repo
                                 --install-deps-from=flathub --state-dir=${CMAKE_CURRENT_BINARY_DIR}
                                 --user --ccache --default-branch=${PROJECT_VERSION}
                                 flatpak-build ${CMAKE_SOURCE_DIR}/packaging/org.sparta.sparta-gui.yml
      COMMAND ${FLATPAK_COMMAND} build-bundle --runtime-repo=https://flathub.org/repo/flathub.flatpakrepo --verbose
                               ${CMAKE_CURRENT_BINARY_DIR}/flatpak-repo
                               ${FLATPAK_BUNDLE} org.sparta.sparta-gui ${PROJECT_VERSION}
      COMMENT "Create Flatpak bundle file of SPARTA-GUI"
      BYPRODUCTS ${FLATPAK_BUNDLE}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  else()
    add_custom_target(flatpak
      COMMAND ${CMAKE_COMMAND} -E echo "The flatpak and flatpak-builder commands required to build a SPARTA-GUI flatpak bundle were not found. Skipping.")
  endif()

  # when compiling on macOS, create an "app bundle"
  if(APPLE)
    # additional targets to populate the bundle tree and create the .dmg image file
    set(APP_CONTENTS ${CMAKE_BINARY_DIR}/sparta-gui.app/Contents)

    # the SPARTA example inputs are bundled into the app so the
    # File -> Open Example menu works from an installed .app; the location
    # can be overridden with -D SPARTA_EXAMPLES_DIR=...
    if(NOT SPARTA_EXAMPLES_DIR)
      set(SPARTA_EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../examples)
    endif()
    set(BUNDLE_EXAMPLES_CMD "")
    if(EXISTS ${SPARTA_EXAMPLES_DIR})
      set(BUNDLE_EXAMPLES_CMD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${SPARTA_EXAMPLES_DIR} ${APP_CONTENTS}/Resources/examples)
    endif()

    add_custom_target(complete-bundle
      ${CMAKE_COMMAND} -E make_directory ${APP_CONTENTS}/bin
      COMMAND ${CMAKE_COMMAND} -E create_symlink ../MacOS/sparta-gui ${APP_CONTENTS}/bin/sparta-gui
      COMMAND ${CMAKE_COMMAND} -E make_directory ${APP_CONTENTS}/Resources
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_README_FILE} ${APP_CONTENTS}/Resources/README.txt
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_ICON_FILE} ${APP_CONTENTS}/Resources
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BACKGROUND_FILE} ${APP_CONTENTS}/Resources
      ${BUNDLE_EXAMPLES_CMD}
      DEPENDS sparta-gui
      COMMENT "Copying additional files into macOS app bundle tree"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    add_custom_target(dmg
      COMMAND ${CMAKE_SOURCE_DIR}/packaging/build_macos_dmg.sh ${PROJECT_VERSION}
      DEPENDS complete-bundle
      COMMENT "Create Drag-n-Drop installer disk image from app bundle"
      BYPRODUCTS SPARTA-GUI-macOS-multiarch-${PROJECT_VERSION}.dmg
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  # settings for packaging Windows NSIS installer on Linux with MinGW cross-compiler.
  # native compilation with MSVC is supported for development only and has no
  # deployment or packaging support; all Windows packages are cross-compiled.
  elseif((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND CMAKE_CROSSCOMPILING)
    set(QT_NO_QTPATHS_DEPLOYMENT_WARNING ON CACHE BOOL "" FORCE)
    add_custom_target(nsis
      COMMAND ${CMAKE_SOURCE_DIR}/packaging/build_windows_cross_nsis.sh ${CMAKE_INSTALL_PREFIX} ${PROJECT_VERSION} ${CMAKE_SOURCE_DIR}
      DEPENDS sparta-gui html pdf
      COMMENT "Create nsis installer with windows binaries"
      BYPRODUCTS SPARTA-GUI-Win10-x86_64-${PROJECT_VERSION}.exe
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  # SPARTA_GUI_USE_PLUGIN is already required by the enclosing if() block
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sparta-gui.desktop DESTINATION ${CMAKE_INSTALL_DATADIR}/applications/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sparta-gui.appdata.xml DESTINATION ${CMAKE_INSTALL_DATADIR}/appdata/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sparta-input.xml DESTINATION ${CMAKE_INSTALL_DATADIR}/mime/packages/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/sparta-input.xml DESTINATION ${CMAKE_INSTALL_DATADIR}/mime/text/x-application-sparta.xml)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/hicolor DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/)
    install(CODE [[
      file(GET_RUNTIME_DEPENDENCIES
        EXECUTABLES $<TARGET_FILE:sparta-gui>
        RESOLVED_DEPENDENCIES_VAR _r_deps
        UNRESOLVED_DEPENDENCIES_VAR _u_deps
      )
      foreach(_file ${_r_deps})
        file(INSTALL
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
          TYPE SHARED_LIBRARY
          FOLLOW_SYMLINK_CHAIN
          FILES "${_file}"
        )
      endforeach()
      list(LENGTH _u_deps _u_length)
      if("${_u_length}" GREATER 0)
        message(WARNING "Unresolved dependencies detected: ${_u_deps}")
      endif() ]]
    )

    add_custom_target(tgz
      COMMAND ${CMAKE_SOURCE_DIR}/packaging/build_linux_tgz.sh ${PROJECT_VERSION}
      DEPENDS sparta-gui
      COMMENT "Create compressed tar file of SPARTA-GUI with dependent libraries and wrapper"
      BYPRODUCTS SPARTA-GUI-Linux-x86_64-${PROJECT_VERSION}.tar.gz
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
endif()

find_package(Git)
if (GIT_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
  add_custom_target(tarball
    COMMAND ${GIT_EXECUTABLE} archive --prefix=sparta-gui-v${PROJECT_VERSION}/
            --output=${CMAKE_BINARY_DIR}/sparta-gui-src-v${PROJECT_VERSION}.tar.gz  HEAD
    COMMENT "Create compressed tar file of the SPARTA-GUI sources"
    BYPRODUCTS sparta-gui-src-v${PROJECT_VERSION}.tar.gz
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()
