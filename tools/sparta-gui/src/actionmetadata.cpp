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

#include "actionmetadata.h"

#include "actionscan.h"

#include <QAction>
#include <QHash>
#include <QString>

namespace {

// Keyed by mnemonic-stripped action text.  Written as what the entry DOES,
// present tense, no trailing period -- these render in the status bar.
const QHash<QString, QString> &tipsByText()
{
    static const QHash<QString, QString> tips = {
        // ---- File
        {"Welcome Screen", "Show the start page with recent files and the bundled examples"},
        {"New Input File", "Start a new, empty SPARTA input script"},
        {"Open Input File", "Open a SPARTA input script in the editor"},
        {"Save Input File", "Save the editor buffer to its file"},
        {"Save Input File As", "Save the editor buffer under a new name"},
        {"View Text File", "Open any text file in a read-only viewer"},
        {"View Image or Movie File(s)...",
         "Open image files in the viewer or import a movie's frames"},
        {"Plot Data File...", "Plot the columns of a text data file as charts"},
        {"Inspect Restart File", "Read a SPARTA restart file and show what it contains"},
        {"Write Restart File...", "Write the current simulation state to a restart file"},
        {"Quit", "Exit SPARTA-GUI"},
        // ---- Edit
        {"Undo", "Undo the last edit"},
        {"Redo", "Redo the last undone edit"},
        {"Copy", "Copy the selection to the clipboard"},
        {"Cut", "Cut the selection to the clipboard"},
        {"Paste", "Paste the clipboard into the editor"},
        {"Insert Snippet...", "Insert a ready-made block of input commands at the cursor"},
        {"Find and Replace...", "Search the editor, optionally replacing matches"},
        {"Preferences...", "Configure SPARTA-GUI: library, fonts, editor, accelerator"},
        {"Reset Preferences to Defaults",
         "Discard all settings and return to the defaults (asks first)"},
        // ---- Run
        {"Run SPARTA from Editor Buffer", "Run the deck as it is in the editor, saved or not"},
        {"Run SPARTA from File", "Save the editor buffer, then run the saved file"},
        {"Stop SPARTA", "Stop the running simulation cleanly at the next timestep"},
        {"Extend Run...", "Continue the finished run by a number of extra steps"},
        {"Check Input", "Check the deck for problems without running it"},
        {"Relaunch SPARTA Instance", "Discard the SPARTA instance and start a fresh one"},
        {"Set Variables...", "Set index variables for the next run (like the -var flag)"},
        {"Insert Restart Commands...",
         "Insert the commands that continue a run from a restart file"},
        {"Create Image", "Render a snapshot of the current simulation state"},
        {"3D Snapshot (VTK)", "Render the current state into the interactive 3D viewer"},
        // ---- Tools
        {"Import Surface (STL / SPARTA)...",
         "Convert an STL or SPARTA surface file and add it to the deck"},
        {"Export to ParaView...", "Convert surface or grid data to ParaView format"},
        {"Surface Quantities Report...",
         "Integrate a per-surface compute or fix into forces, moments and heat flux"},
        {"Parametric Sweep...", "Run the deck repeatedly while varying index variables"},
        {"Run History...", "Browse the archived output of earlier runs"},
        // ---- View
        {"Run Workspace", "Show the deck beside its output (Ctrl+1)"},
        {"Analyze Workspace", "Give the window to the charts (Ctrl+2)"},
        {"Visualize Workspace", "Give the window to the rendered views (Ctrl+3)"},
        {"Output Window", "Show or hide the captured console output"},
        {"Charts Window", "Show or hide the live stats charts"},
        {"Viewer Window", "Show or hide the image/slide-show/3D viewer"},
        {"Variables Window", "Show or hide the index-variables panel"},
        {"Parametric Sweep Window", "Show or hide the parametric-sweep panel"},
        {"Run History Window", "Show or hide the run-history panel"},
        {"Diagnostics Window", "Show or hide the input-check findings"},
        {"Project Files Window", "Show or hide the files beside the current deck"},
        {"Snapshot in Viewer", "Bring the rendered snapshot to the front of the viewer"},
        {"Slide Show in Viewer", "Bring the run's image frames to the front of the viewer"},
        {"3D Scene in Viewer", "Bring the interactive 3D scene to the front of the viewer"},
        {"3D Viewer Window (VTK)", "Open the interactive 3D scene in its own window"},
        {"Reset Layout", "Restore this workspace's default panel arrangement"},
        // ---- About
        {"About SPARTA-GUI", "Version, credits and license of SPARTA-GUI"},
        {"Quick Help", "A short guide to running and visualizing a deck"},
        {"SPARTA-GUI Documentation", "Open the SPARTA-GUI manual in the browser"},
        {"SPARTA Online Manual", "Open the SPARTA manual in the browser"},
        {"Check for SPARTA update", "Check for and download a newer SPARTA shared library"},
    };
    return tips;
}

} // namespace

void applyActionMetadata(const QMenuBar *bar)
{
    const auto infos = scanMenuBar(bar);
    for (const auto &info : infos) {
        if (!info.action) continue;
        QString tip = tipsByText().value(info.text);
        // dynamic entries have data for text; describe them by where they live
        if (tip.isEmpty()) {
            if (info.path.startsWith(QLatin1String("File > Open Example")))
                tip = QStringLiteral(
                    "Open this bundled example (copied somewhere writable if needed)");
            else if (info.path == QLatin1String("File") &&
                     info.action->objectName().startsWith(QLatin1String("recent")))
                tip = QStringLiteral("Reopen this recently used file");
            else if (info.text.endsWith(QLatin1Char('.')) && info.path == QLatin1String("File"))
                tip = QStringLiteral("Reopen this recently used file");
        }
        if (tip.isEmpty()) continue; // the coverage test reports these, not us
        info.action->setStatusTip(tip);
        // the same line serves as the hover text once actions sit in a toolbar
        if (info.action->toolTip() == info.action->text() ||
            info.action->toolTip() == info.text)
            info.action->setToolTip(tip);
    }
}

// Local Variables:
// c-basic-offset: 4
// End:
