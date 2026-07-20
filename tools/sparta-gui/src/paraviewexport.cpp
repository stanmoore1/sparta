/* ----------------------------------------------------------------------
   SPARTA-GUI - a graphical user interface for SPARTA

   Command builders for the bundled ParaView conversion scripts.  See
   paraviewexport.h for the rationale (these run under pvpython; the GUI
   shells out to them rather than reimplementing their VTK pipelines).
------------------------------------------------------------------------- */

#include "paraviewexport.h"

#include <QFileInfo>

namespace ParaviewExport {

QString scriptName(Mode mode)
{
    return mode == Mode::Grid ? QStringLiteral("grid2paraview.py")
                              : QStringLiteral("surf2paraview.py");
}

QStringList buildScriptArgs(const Settings &s, const QString &scriptPath)
{
    QStringList args;
    args << scriptPath << s.inputFile << s.outputName;

    // optional dump result files (already glob-expanded by the caller)
    if (!s.resultFiles.isEmpty()) {
        args << QStringLiteral("-r");
        args << s.resultFiles;
    }

    if (s.mode == Mode::Surface) {
        if (s.exodus) args << QStringLiteral("-e");
    } else {
        // grid chunk sizes: emit only when they differ from the script default
        // of 100, keeping the command line minimal and deterministic
        if (s.xchunk != 100) args << QStringLiteral("-x") << QString::number(s.xchunk);
        if (s.ychunk != 100) args << QStringLiteral("-y") << QString::number(s.ychunk);
        if (s.zchunk != 100) args << QStringLiteral("-z") << QString::number(s.zchunk);
    }
    return args;
}

QString expectedOutput(const Settings &s)
{
    if (s.mode == Mode::Surface && s.exodus) return s.outputName + QStringLiteral(".ex2");
    return s.outputName + QStringLiteral(".pvd");
}

bool validate(const Settings &s, QString &err)
{
    err.clear();
    if (s.inputFile.trimmed().isEmpty()) {
        err = QStringLiteral("No input file selected.");
        return false;
    }
    if (!QFileInfo::exists(s.inputFile)) {
        err = QStringLiteral("Input file does not exist:\n%1").arg(s.inputFile);
        return false;
    }
    if (s.outputName.trimmed().isEmpty()) {
        err = QStringLiteral("No ParaView output name given.");
        return false;
    }
    if (s.mode == Mode::Grid) {
        if (s.xchunk < 1 || s.ychunk < 1 || s.zchunk < 1) {
            err = QStringLiteral("Grid chunk sizes must be positive integers.");
            return false;
        }
    }
    return true;
}

} // namespace ParaviewExport

// Local Variables:
// c-basic-offset: 4
// End:
