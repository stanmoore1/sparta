#!/bin/bash

# Build a drag-n-drop installer .dmg from the sparta-gui.app bundle, styled with
# the SPARTA icon, a background image, a README and an /Applications alias.
#
# Two modes:
#   default        - deploy Qt with `macdeployqt -dmg` (universal, non-VTK build),
#                    then style the image.  Backwards compatible: called with just
#                    the version as $1 it behaves exactly as before.
#   DMG_PREBUNDLED - the app is already fully bundled AND code-signed (e.g. a VTK
#     =yes          build that bundled VTK with dylibbundler and re-signed).  Do
#                    not run macdeployqt, do not copy/modify anything inside the
#                    signed bundle (that would invalidate the signature): build
#                    the image directly from the app and take the README and
#                    background from this script's directory instead.
#
# Environment:
#   SPARTA_PLUGIN_LIB  path to the libsparta dylib to bundle (default mode only)
#   DMG_PREBUNDLED     "yes" to use the pre-bundled/signed path
#   DMG_OUTPUT         output .dmg filename (default SPARTA-GUI-macOS-multiarch-v<ver>.dmg)

APP_NAME=sparta-gui
VERSION="$1"
DMG_PREBUNDLED="${DMG_PREBUNDLED:-no}"
DMG_OUTPUT="${DMG_OUTPUT:-SPARTA-GUI-macOS-multiarch-v${VERSION}.dmg}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Delete old files, if they exist"
rm -f ${APP_NAME}.dmg ${APP_NAME}-rw.dmg "${DMG_OUTPUT}"

if [ "${DMG_PREBUNDLED}" = "yes" ]; then
    # -------- pre-bundled + signed app: build the image without touching it -----
    echo "Create initial writable dmg directly from the pre-bundled app"
    appmb=$(du -sm ${APP_NAME}.app | awk '{print $1}')
    sizemb=$(( appmb + 150 ))
    hdiutil create -volname "${APP_NAME}" -srcfolder ${APP_NAME}.app \
        -fs HFS+ -format UDRW -ov -size ${sizemb}m ${APP_NAME}-rw.dmg
else
    # -------- default: deploy Qt into the app and let macdeployqt make the dmg ---
    rm -f ${APP_NAME}.app/Contents/Frameworks/libsparta.0.dylib

    # bundle the SPARTA shared library if this is a plugin-mode SPARTA-GUI binary.
    # the library to bundle must be passed in the SPARTA_PLUGIN_LIB environment
    # variable (e.g. a universal libsparta dylib built alongside the GUI)
    if $(./${APP_NAME}.app/Contents/MacOS/sparta-gui -h | grep -q pluginpath); then
        if [ -z "${SPARTA_PLUGIN_LIB}" ] || [ ! -f "${SPARTA_PLUGIN_LIB}" ]; then
            echo "ERROR: set SPARTA_PLUGIN_LIB to the path of the SPARTA shared library to bundle"
            exit 1
        fi
        mkdir -p ${APP_NAME}.app/Contents/Frameworks
        cp "${SPARTA_PLUGIN_LIB}" ${APP_NAME}.app/Contents/Frameworks/libsparta.0.dylib
        chmod 0755 ${APP_NAME}.app/Contents/Frameworks/libsparta.0.dylib
    fi

    # The pre-compiled SPARTA library is built with OpenMP and records a dependency
    # on /usr/local/lib/libomp.dylib.  Apple ships no libomp, so on hosts that were
    # set up without an OpenMP runtime (e.g. CI runners) macdeployqt cannot resolve
    # or bundle it and prints a cryptic otool "can't open file" error.  Detect that
    # case so the bundle is built cleanly without OpenMP support instead.
    SKIP_LIBOMP=no
    SPARTA_LIB=${APP_NAME}.app/Contents/Frameworks/libsparta.0.dylib
    if [ -f "${SPARTA_LIB}" ] \
       && otool -L "${SPARTA_LIB}" 2>/dev/null | grep -q '/usr/local/lib/libomp.dylib' \
       && [ ! -e /usr/local/lib/libomp.dylib ]; then
        echo "NOTE: libsparta references libomp.dylib but no OpenMP runtime is installed;"
        echo "      building the bundle without OpenMP support and skipping libomp deployment."
        SKIP_LIBOMP=yes
    fi

    echo "Create initial dmg file with macdeployqt"
    if [ "${SKIP_LIBOMP}" = "yes" ]; then
        # drop the expected, harmless errors about the unresolved libomp dependency
        macdeployqt ${APP_NAME}.app -dmg 2>&1 | grep -v 'libomp\.dylib'
    else
        macdeployqt ${APP_NAME}.app -dmg
    fi
    echo "Create writable dmg file"
    hdiutil convert ${APP_NAME}.dmg -format UDRW -o ${APP_NAME}-rw.dmg
fi

echo "Mount writeable DMG file in read-write mode. Keep track of device and volume names"
DEVICE=$(hdiutil attach -readwrite -noverify ${APP_NAME}-rw.dmg | grep '^/dev/' | sed 1q | awk '{print $1}')
VOLUME=$(df | grep ${DEVICE} | sed -e 's/^.*\(\/Volumes\/\)/\1/')
sleep 2

echo "Create link to Application folder and place README and background image files"

pushd "${VOLUME}"
ln -s /Applications .
mkdir .background
if [ "${DMG_PREBUNDLED}" = "yes" ]; then
    # take these from the source tree so the signed app bundle is not modified
    cp "${SCRIPT_DIR}/README.macos" README.txt
    cp "${SCRIPT_DIR}/SPARTA_DMG_Background.png" .background/background.png
else
    mv ${APP_NAME}.app/Contents/Resources/README.txt .
    mv ${APP_NAME}.app/Contents/Resources/SPARTA_DMG_Background.png .background/background.png
fi
mv ${APP_NAME}.app SPARTA-GUI.app

# Attach the icon to the executable/lib only in the default (unsigned) path;
# doing it to a signed bundle would invalidate the code signature.  A signed
# app already shows its icon via CFBundleIconFile in Info.plist.
if [ "${DMG_PREBUNDLED}" != "yes" ]; then
    cd SPARTA-GUI.app/Contents
    echo "Attach icons to SPARTA-GUI executable and lib"
    echo "read 'icns' (-16455) \"Resources/sparta-gui.icns\";" > icon.rsrc
    Rez -a icon.rsrc -o MacOS/sparta-gui
    SetFile -a C MacOS/sparta-gui
    if [ -f Frameworks/libsparta.0.dylib ]; then
        Rez -a icon.rsrc -o Frameworks/libsparta.0.dylib
        SetFile -a C Frameworks/libsparta.0.dylib
    fi
    rm icon.rsrc
    cd "${VOLUME}"
fi
popd

echo 'Tell the Finder to resize the window, set the background,'
echo 'change the icon size, place the icons in the right position, etc.'
echo '
    tell application "Finder"
    tell disk "'${APP_NAME}'"

      -- wait for the image to finish mounting
      set open_attempts to 0
      repeat while open_attempts < 4
        try
          open
            delay 1
            set open_attempts to 5
          close
        on error errStr number errorNumber
          set open_attempts to open_attempts + 1
          delay 10
        end try
      end repeat
      delay 5

      -- open the image the first time and save a .DS_Store
      -- just the background and icon setup
      open
        set current view of container window to icon view
        set theViewOptions to the icon view options of container window
        set background picture of theViewOptions to file ".background:background.png"
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 64
        delay 5
      close

      -- next set up the position of the app and Applications symlink
      -- plus hide all window decorations
      open
        update without registering applications
        tell container window
          set sidebar width to 0
          set statusbar visible to false
          set toolbar visible to false
          set the bounds to { 100, 40, 868, 640 }
          set position of item "'SPARTA-GUI'.app" to { 190, 216 }
          set position of item "Applications" to { 576, 216 }
          set position of item "README.txt" to { 190, 400 }
        end tell
        update without registering applications
        delay 5
      close

      -- one last open and close to check the results
      open
        delay 5
      close
    end tell
    delay 1
  end tell
' | osascript

sync

echo "Unmount modified disk image and convert to compressed read-only image"
hdiutil detach "${DEVICE}" || hdiutil detach "${DEVICE}" -force || true
hdiutil convert "${APP_NAME}-rw.dmg" -format UDZO -o "${DMG_OUTPUT}"

echo "Attach icon to .dmg file"
echo "read 'icns' (-16455) \"sparta-gui.app/Contents/Resources/sparta-gui.icns\";" > icon.rsrc
Rez -a icon.rsrc -o "${DMG_OUTPUT}"
SetFile -a C "${DMG_OUTPUT}"
rm icon.rsrc

echo "Delete temporary disk images"
rm -f "${APP_NAME}-rw.dmg"
rm -f "${APP_NAME}.dmg"

exit 0
