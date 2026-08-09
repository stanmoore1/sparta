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

#ifndef LIBRARYACQUIRE_H
#define LIBRARYACQUIRE_H

#include <QString>
#include <QStringList>

class QWidget;

/**
 * @brief Finding, naming and downloading the SPARTA shared library.
 *
 * Three places acquire the library -- the first run with none configured,
 * the Preferences button, and Check for Update -- and each of them used to
 * re-derive the same things: where a downloaded copy is kept, what a library
 * is called on this platform, which glob a file dialog should filter on, and
 * the rule that a cancelled download is a choice rather than a failure to
 * report.  Four copies of the platform ifdef chain had already drifted into
 * three slightly different file-dialog patterns.
 *
 * These are the pieces they share.  Nothing here loads a library or touches
 * settings: that is the caller's, because what to do on success differs (the
 * first run continues into the loaded library, the other two relaunch).
 */
namespace LibraryAcquire {

/// @brief The filter string for a file dialog offering SPARTA libraries.
[[nodiscard]] QString fileDialogPattern();

/// @brief The glob patterns a SPARTA shared library matches on this platform.
[[nodiscard]] QStringList fileNamePatterns();

/**
 * @brief Directories worth searching for a library, most specific first.
 *
 * The current directory, then the dynamic loader's path, then the places a
 * downloaded or system-installed copy lands.
 */
[[nodiscard]] QStringList searchPaths();

/**
 * @brief Every library-shaped file on this system, canonical and deduplicated.
 *
 * In @ref searchPaths order, so a caller trying each in turn tries the most
 * specific first.  Says nothing about whether any of them will load.
 */
[[nodiscard]] QStringList candidates();

/**
 * @brief Where a downloaded library is kept, creating the directory.
 * @return the full path to write to, or an empty string if it cannot be created
 */
[[nodiscard]] QString downloadDestination();

/**
 * @brief Does this file name look like a SPARTA shared library?
 *
 * A heuristic over the name only -- whether the file loads is the real test,
 * and a user is allowed to overrule this.
 */
[[nodiscard]] bool nameLooksRight(const QString &path);

/// @brief What became of a download.
enum class Result {
    Ok,        ///< the file is on disk at the requested path
    Cancelled, ///< the user stopped it; a choice, so nothing to report
    Failed,    ///< it went wrong; @c error says how
};

/**
 * @brief Download this platform's pre-compiled library to @p dest.
 * @param parent parent for the progress dialog
 * @param dest   where to write it (from @ref downloadDestination)
 * @param error  set to the failure reason when the result is Failed
 *
 * Keeps a backup of anything already at @p dest and puts it back if the
 * transfer fails, because @p dest may be the very library the running process
 * has loaded.
 */
Result download(QWidget *parent, const QString &dest, QString *error);

} // namespace LibraryAcquire

#endif // LIBRARYACQUIRE_H

// Local Variables:
// c-basic-offset: 4
// End:
