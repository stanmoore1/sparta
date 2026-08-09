// -*- c++ -*- /////////////////////////////////////////////////////////////////////////
// SPARTA-GUI - A Graphical Tool to Learn and Explore the SPARTA DSMC Simulation Software
//
// Copyright (c) 2023, 2024, 2025, 2026  Axel Kohlmeyer
//
// Documentation: https://sparta.github.io/sparta-gui/
// Contact: akohlmey@gmail.com
//
// This software is distributed under the GNU General Public License version 2 or later.
////////////////////////////////////////////////////////////////////////////////////////

#include "libraryacquire.h"

#include "helpers.h"
#include "urldownloader.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

namespace LibraryAcquire {

QString fileDialogPattern()
{
#if defined(Q_OS_MACOS)
    return QStringLiteral("SPARTA shared library (libsparta*.dylib)");
#elif defined(Q_OS_WIN32)
    return QStringLiteral("SPARTA shared library (libsparta*.dll)");
#else
    return QStringLiteral("SPARTA shared library (libsparta*.so*)");
#endif
}

QStringList fileNamePatterns()
{
#if defined(Q_OS_MACOS)
    return {QStringLiteral("libsparta*.dylib")};
#elif defined(Q_OS_WIN32)
    return {QStringLiteral("libsparta*.dll")};
#else
    return {QStringLiteral("libsparta*.so*")};
#endif
}

QStringList searchPaths()
{
    QStringList dirs{QStringLiteral(".")};

#if defined(Q_OS_MACOS)
    dirs.append(QString::fromLocal8Bit(qgetenv("DYLD_LIBRARY_PATH")).split(":", Qt::SkipEmptyParts));
    // the library may travel inside an application bundle
    dirs.append(QCoreApplication::applicationDirPath() + "/../Frameworks");
    dirs.append({"/Applications/SPARTA-GUI.app/Contents/Frameworks",
                 "/Applications/SPARTA.app/Contents/Frameworks"});
#elif defined(Q_OS_WIN32)
    dirs.append(QString::fromLocal8Bit(qgetenv("PATH")).split(";", Qt::SkipEmptyParts));
#else
    dirs.append(QString::fromLocal8Bit(qgetenv("LD_LIBRARY_PATH")).split(":", Qt::SkipEmptyParts));
#endif

    // where a previously downloaded copy was put
    dirs.append(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation));
    // and the usual system locations; these simply do not exist elsewhere
    dirs.append({"/usr/lib", "/usr/lib64", "/lib/x86_64-linux-gnu", "/usr/local/lib",
                 "/usr/local/lib64"});

    return dirs;
}

QStringList candidates()
{
    const QStringList filter = fileNamePatterns();

    QStringList found;
    for (const auto &dir : searchPaths())
        for (const auto &entry : QDir(dir).entryInfoList(filter))
            found.append(entry.canonicalFilePath());

    found.removeAll(QString());
    found.removeDuplicates();
    return found;
}

QString downloadDestination()
{
    // the same directory QSettings keeps preferences in, so a downloaded
    // library travels with the configuration that points at it
    const QString configDir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    if (configDir.isEmpty() || !QDir().mkpath(configDir)) return {};
    return configDir + QDir::separator() + getSpartaLibName();
}

bool nameLooksRight(const QString &path)
{
    return path.contains(QStringLiteral("libsparta"), Qt::CaseInsensitive);
}

Result download(QWidget *parent, const QString &dest, QString *error)
{
    const QString url = getSpartaDownloadUrl();
    if (url.isEmpty()) {
        if (error)
            *error = QStringLiteral("No pre-compiled SPARTA library is offered for this build.");
        return Result::Failed;
    }

    URLDownloader downloader(parent);
    // keepBackup: dest may be the very library this process has already loaded
    if (downloader.download(url, dest, true, true)) return Result::Ok;

    // Cancelling is a choice and says nothing, so it is not a failure to
    // report; every caller had to know that and one of them used to get it
    // wrong in a different way each time.
    if (downloader.wasAborted()) return Result::Cancelled;

    if (error) *error = downloader.errorString();
    return Result::Failed;
}

} // namespace LibraryAcquire

// Local Variables:
// c-basic-offset: 4
// End:
