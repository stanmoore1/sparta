#!/bin/bash

APP_NAME=sparta-gui
DESTDIR=${PWD}/SPARTA_GUI
SYSROOT="$1"
VERSION="$2"
SRCDIR="$3"

echo "Delete old files, if they exist"
rm -rvf ${DESTDIR}/SPARTA_GUI ${DESTDIR}/SPARTA-Win10-amd64*.exe

echo "Create staging area for deployment and populate"
DESTDIR=${DESTDIR} cmake --install .  --prefix "/"

# no static libs needed
rm -rvf ${DESTDIR}/lib ${DESTDIR}/bin/libsparta.dll
# provide the SPARTA library DLL: bundle a locally cross-built one if
# SPARTA_PLUGIN_LIB points at it (as the macOS/Linux packagers do), else
# download a pre-compiled basic library.
if [ -n "${SPARTA_PLUGIN_LIB}" ] && [ -f "${SPARTA_PLUGIN_LIB}" ]
then \
    echo "Bundle locally-built SPARTA library DLL: ${SPARTA_PLUGIN_LIB}"
    cp -v "${SPARTA_PLUGIN_LIB}" ${DESTDIR}/bin/libsparta.dll
else \
    wget https://sparta.github.io/sparta-gui/libsparta.dll
    mv -v libsparta.dll ${DESTDIR}/bin/
fi
# ffmpeg (movie export) and gzip (compressed dumps) are optional runtime
# helpers; bundle them if available but do not fail the build if not.
if wget -q https://sparta.github.io/thirdparty/ffmpeg-win64.exe.gz
then \
    gunzip ffmpeg-win64.exe.gz && mv ffmpeg-win64.exe ${DESTDIR}/bin/ffmpeg.exe
else \
    echo "NOTE: ffmpeg-win64 not available; movie export will be unavailable"
fi
if wget -q https://sparta.github.io/thirdparty/gzip.exe.gz
then \
    gunzip gzip.exe.gz && mv gzip.exe ${DESTDIR}/bin/
else \
    echo "NOTE: gzip.exe not available; compressed-dump support will be unavailable"
fi

skipdlls="msvcrt ADVAPI32 CFGMGR32 GDI32 KERNEL32 MPR NETAPI32 PSAPI SHELL32 USER32 USERENV UxTheme VERSION WS2_32 WSOCK32 d3d11 dwmapi libsparta msvcrt_ole32 dxgi IMM32 ole32 OLEAUT32 WINMM WTSAPI32 COMCTL32 PSAPI bcrypt CRYPT32 IPHLPAPI Secur32 api-ms-win-core-path-l1-1-0 WLDAP32 api-ms-win-core-synch-l1-2-0 AUTHZ d3d12 DWrite ntdll api-ms-win-core-winrt-l1-1-0 api-ms-win-core-winrt-string-l1-1-0 comdlg32 d2d1 d3d9 SETUPAPI SHCORE SHLWAPI DNSAPI WINHTTP ncrypt"
echo "Copying required DLL files"
for dll in $(objdump -p *.exe | sed -n -e '/DLL Name:/s/^.*DLL Name: *//p' | sort | uniq)
do \
    doskip=0
    for skip in ${skipdlls}
    do \
        test ${dll} = ${skip}.dll && doskip=1
        test ${dll} = ${skip}.DLL && doskip=1
    done
    test ${doskip} -eq 1 && continue
    test -f ${DESTDIR}/bin/${dll} || cp -v ${SYSROOT}/bin/${dll} ${DESTDIR}/bin || exit 1
done

echo "Copy required Qt plugins"
mkdir -p ${DESTDIR}/qt6plugins
for plugin in imageformats platforms styles tls
do \
    cp -r ${SYSROOT}/lib/qt6/plugins/${plugin} ${DESTDIR}/qt6plugins/
done

echo "Check dependencies of DLL files"
for dll in $(objdump -p ${DESTDIR}/bin/*.dll ${DESTDIR}/qt6plugins/*/*.dll | sed -n -e '/DLL Name:/s/^.*DLL Name: *//p' | sort | uniq)
do \
    doskip=0
    for skip in ${skipdlls}
    do \
        test ${dll} = ${skip}.dll && doskip=1
        test ${dll} = ${skip}.DLL && doskip=1
    done
    test ${doskip} -eq 1 && continue
    test -f ${DESTDIR}/bin/${dll} || cp -v ${SYSROOT}/bin/${dll} ${DESTDIR}/bin || exit 1
done

for dll in $(objdump -p ${DESTDIR}/bin/*.dll ${DESTDIR}/qt6plugins/*/*.dll | sed -n -e '/DLL Name:/s/^.*DLL Name: *//p' | sort | uniq)
do \
    doskip=0
    for skip in ${skipdlls}
    do \
        test ${dll} = ${skip}.dll && doskip=1
        test ${dll} = ${skip}.DLL && doskip=1
    done
    test ${doskip} -eq 1 && continue
    test -f ${DESTDIR}/bin/${dll} || cp -v ${SYSROOT}/bin/${dll} ${DESTDIR}/bin || exit 1
done

cat > ${DESTDIR}/bin/qt.conf <<EOF
[Paths]
Plugins = ../qt6plugins
EOF

# bundle the manual PDF if it was built; the NSIS script globs *.pdf so at
# least one must be present in the staging dir (the workflow drops in a small
# placeholder that points at the online manual when the full PDF is not built)
if [ -f sparta-gui-v${VERSION}.pdf ]
then \
    cp -v sparta-gui-v${VERSION}.pdf ${DESTDIR}/SPARTA-GUI-Manual.pdf
fi
cp -v ${SRCDIR}/LICENSE ${DESTDIR}/LICENSE.txt
unix2dos ${DESTDIR}/LICENSE.txt
cp -v ${SRCDIR}/packaging/sparta-gui.nsis ${SRCDIR}/packaging/FileAssociation.nsh ${DESTDIR}
cp -v ${SRCDIR}/resources/sparta-gui.ico ${SRCDIR}/resources/icons/sparta-gui-banner.bmp ${DESTDIR}
revflag=$(git rev-parse --abbrev-ref HEAD)
pushd ${DESTDIR}
makensis -DMINGW="${SYSROOT}/bin/" -DVERSION="${VERSION}" -DBIT=64 -DLMPREV="${revflag}" \
         sparta-gui.nsis
popd
