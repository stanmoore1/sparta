#!/bin/bash

APP_NAME=sparta-gui
DESTDIR=${PWD}/../SPARTA_GUI
VERSION="$1"

echo "Delete old files, if they exist"
rm -rf ${DESTDIR} ../SPARTA-GUI-Linux-x86_64*.tar.gz

echo "Create staging area for deployment and populate"
DESTDIR=${DESTDIR} cmake --install .  --prefix "/"

echo "Remove debug info"
for s in ${DESTDIR}/bin/* ${DESTDIR}/lib/libsparta*
do \
    test -f $s && strip --strip-debug $s
done

# download pre-compiled SPARTA shared library
if $(LD_LIBRARY_PATH=${DESTDIR}/lib ${DESTDIR}/bin/sparta-gui -h | grep -q pluginpath)
then \
    echo "Download basic SPARTA shared library"
    mkdir -p ${DESTDIR}/libexec/sparta
    curl -L -o ${DESTDIR}/libexec/sparta/libsparta.so.0 https://sparta.github.io/sparta-gui/libsparta.so.0
    chmod +x ${DESTDIR}/libexec/sparta/libsparta.so.0
fi

echo "Remove libc, gcc, and X11 related shared libs"
rm -f ${DESTDIR}/lib/ld*.so ${DESTDIR}/lib/ld*.so.[0-9]
rm -f ${DESTDIR}/lib/lib{c,dl,rt,m,pthread}.so.?
rm -f ${DESTDIR}/lib/lib{c,dl,rt,m,pthread}-[0-9].[0-9]*.so
rm -f ${DESTDIR}/lib/libX* ${DESTDIR}/lib/libxcb* ${DESTDIR}/lib/libxkb*
rm -f ${DESTDIR}/lib/libgcc_s*
rm -f ${DESTDIR}/lib/libstdc++*

# get Qt dir
QTDIR=$(ldd ${DESTDIR}/bin/sparta-gui | grep libQt.Core | sed -e 's/^.*=> *//' -e 's/libQt\(.\)Core.so.*$/qt\1/')

# configure some settings files for Qt
cat > ${DESTDIR}/bin/qt.conf <<EOF
[Paths]
Plugins = ../qtplugins
EOF

cat > ${DESTDIR}/bin/qtlogging.ini <<EOF
[Rules]
*.debug=false
qt.qpa.xcb.xcberror.warning=false
EOF

# platform plugin
mkdir -p ${DESTDIR}/qtplugins/platforms
cp ${QTDIR}/plugins/platforms/libqxcb.so ${DESTDIR}/qtplugins/platforms
chmod +x ${DESTDIR}/qtplugins/platforms/libqxcb.so

# get platform plugin dependencies
QTDEPS=$(LD_LIBRARY_PATH=${DESTDIR}/lib ldd ${QTDIR}/plugins/platforms/libqxcb.so | grep -v ${DESTDIR} | grep libQt[56] | sed -e 's/^.*=> *//' -e 's/\(libQt[56].*.so.*\) .*$/\1/')
for dep in ${QTDEPS}
do \
    cp ${dep} ${DESTDIR}/lib
    chmod +x ${DESTDIR}/lib/${dep}
done

echo "Add additional plugins for Qt"
for dir in styles imageformats tls
do \
    cp -r  ${QTDIR}/plugins/${dir} ${DESTDIR}/qtplugins/
    chmod +x ${DESTDIR}/qtplugins/*/*.so
done

# get imageplugin dependencies
for s in ${DESTDIR}/qtplugins/imageformats/*.so
do \
    QTDEPS=$(LD_LIBRARY_PATH=${DESTDIR}/lib ldd $s | grep -v ${DESTDIR} | grep -E '(libQt.|jpeg)' | sed -e 's/^.*=> *//' -e 's/\(lib.*.so.*\) .*$/\1/')
    for dep in ${QTDEPS}
    do \
        cp ${dep} ${DESTDIR}/lib
        chmod +x ${DESTDIR}/lib/${dep}
    done
done

echo "Set up wrapper script"
MYDIR=$(dirname "$0")
cp ${MYDIR}/xdg-open ${DESTDIR}/bin
cp ${MYDIR}/linux_wrapper.sh ${DESTDIR}/bin
chmod 0755 ${DESTDIR}/bin/linux_wrapper.sh
for s in ${DESTDIR}/bin/*
do \
        EXE=$(basename $s)
        test ${EXE} = linux_wrapper.sh && continue
        test ${EXE} = qt.conf && continue
        test ${EXE} = qtlogging.ini && continue
        test ${EXE} = xdg-open && continue
        ln -s bin/linux_wrapper.sh ${DESTDIR}/${EXE}
done

pushd ..
tar -czvvf SPARTA-GUI-Linux-x86_64-v${VERSION}.tar.gz SPARTA_GUI
popd
mv ../SPARTA-GUI-Linux-x86_64-v${VERSION}.tar.gz .

echo "Cleanup dir"
rm -r ${DESTDIR}
exit 0
