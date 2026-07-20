/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Pure, GUI-free helpers that build the command lines for the bundled
   ParaView conversion scripts (tools/paraview/surf2paraview.py and
   grid2paraview.py) and validate their settings.  These scripts require
   ParaView's pvpython/pvbatch interpreter (they import vtk); the GUI does
   not reimplement them, it locates pvpython and runs them as subprocesses.

   Kept free of Qt widgets and SPARTA so it can be unit-tested in isolation,
   mirroring stlimport.{h,cpp} and dumpimage.{h,cpp}.
------------------------------------------------------------------------- */

#ifndef PARAVIEWEXPORT_H
#define PARAVIEWEXPORT_H

#include <QString>
#include <QStringList>

namespace ParaviewExport {

/** @brief Which SPARTA-to-ParaView converter to run. */
enum class Mode {
    Surface, ///< surf2paraview.py: surface geometry (+ optional per-surf dumps)
    Grid     ///< grid2paraview.py: grid cells (+ optional per-grid dumps)
};

/**
 * @brief Plain-data settings describing one conversion run.
 *
 * @c resultFiles is the list of SPARTA dump result files to associate with
 * the geometry over time (the scripts' @c -r option).  The GUI expands any
 * user-typed glob into concrete paths before filling this in, because
 * QProcess does not perform shell globbing.
 */
struct Settings {
    Mode mode = Mode::Surface;
    QString inputFile;         ///< surf file (Surface) or grid-description file (Grid)
    QString outputName;        ///< ParaView output base name (no extension)
    QStringList resultFiles;   ///< optional, already-expanded dump result files (-r)
    bool exodus = false;       ///< Surface only: write Exodus II (.ex2) instead of .pvd (-e)
    int xchunk = 100;          ///< Grid only: x grid chunk size (-x)
    int ychunk = 100;          ///< Grid only: y grid chunk size (-y)
    int zchunk = 100;          ///< Grid only: z grid chunk size (-z)
};

/** @brief Bundled script file name for a mode (e.g. "surf2paraview.py"). */
QString scriptName(Mode mode);

/**
 * @brief Build the argument vector passed to pvpython for a conversion.
 *
 * Returns @c {scriptPath, inputFile, outputName, [extra options...]} so the
 * caller can start @c QProcess(pvpython, buildScriptArgs(...)).  Only
 * non-default options are emitted, in a deterministic order, matching the
 * "omit defaults" contract of the other command builders.
 *
 * @param s          the conversion settings
 * @param scriptPath full path to the bundled .py script
 */
QStringList buildScriptArgs(const Settings &s, const QString &scriptPath);

/**
 * @brief The primary ParaView file produced by a successful run.
 *
 * Both converters write @c <outputName>.pvd as their main collection file
 * (Exodus writes @c <outputName>.ex2).  This is what the GUI opens in
 * ParaView afterwards.
 */
QString expectedOutput(const Settings &s);

/**
 * @brief Validate settings before running.
 * @param s   the settings to check
 * @param err set to a human-readable reason when the result is false
 * @return true when the settings are runnable
 */
bool validate(const Settings &s, QString &err);

} // namespace ParaviewExport

#endif // PARAVIEWEXPORT_H
