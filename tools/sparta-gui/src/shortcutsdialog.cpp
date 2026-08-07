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

#include "shortcutsdialog.h"

#include "actionscan.h"

#include <QAction>
#include <QDialogButtonBox>
#include <QKeySequence>
#include <QMenuBar>
#include <QTabWidget>
#include <QTextBrowser>
#include <QVBoxLayout>

namespace {

// The first page: what used to be Quick Help, reshaped from one wall of text
// into sections named after what the user is trying to do.
const char *GETTING_STARTED = R"(
<h3>Run a simulation</h3>
<p>The main window is a text editor for SPARTA input scripts. Press
<b>Ctrl+Enter</b> (Run &gt; Run SPARTA from Editor Buffer) to run the deck
exactly as it is in the editor. The console output appears in the Output
panel, the stats columns are charted live, and a progress bar tracks the run.
Stop cleanly at the next timestep with <b>Ctrl+/</b>. A finished run can be
continued with Run &gt; Extend Run (<b>Ctrl+E</b>).</p>

<h3>Start from an example</h3>
<p>File &gt; Open Example lists every input script bundled with SPARTA, and
the Welcome screen (<b>Alt+Home</b>) shows them as a gallery. Opening a file
switches the working directory to its folder, so the deck finds its surface
and data files; the current directory is shown in the status bar.</p>

<h3>See results</h3>
<p>The window is organized into three workspaces, switched with
<b>Ctrl+1</b>-<b>Ctrl+3</b>: <i>Run</i> shows the deck beside its output,
<i>Analyze</i> gives the window to the charts, <i>Visualize</i> to the
rendered views. While nothing is running, <b>Ctrl+I</b> renders a snapshot
of the current state; a deck containing a <code>dump image</code> command
feeds the slide show as frames are written.</p>

<h3>Check before running</h3>
<p><b>Ctrl+K</b> checks the deck without running it: unknown commands,
wrong argument counts, missing files and undefined references are marked in
the editor and listed in the Diagnostics panel. The documentation for the
command on the current line is one <b>Ctrl+?</b> away.</p>

<h3>Find anything</h3>
<p>The command palette (<b>Ctrl+Shift+P</b>) searches every menu action by
name and shows its shortcut. The full shortcut list is on the next tab of
this dialog (<b>F1</b>).</p>

<h3>Good to know</h3>
<p>Preferences (<b>Ctrl+P</b>) selects the SPARTA shared library, fonts, and
the KOKKOS accelerator package with its thread count. As a graphical
application, SPARTA-GUI cannot run SPARTA in parallel with MPI. Files can
also be opened by command-line argument or drag-and-drop.</p>
)";

} // namespace

ShortcutsDialog::ShortcutsDialog(QMenuBar *bar, QWidget *parent) : QDialog(parent), menubar(bar)
{
    setObjectName("shortcutsdialog");
    setWindowTitle("SPARTA-GUI - Help");
    resize(640, 560);

    auto *layout = new QVBoxLayout(this);
    tabs         = new QTabWidget(this);
    tabs->setObjectName("helptabs");

    auto *started = new QTextBrowser(this);
    started->setObjectName("gettingstarted");
    started->setOpenExternalLinks(true);
    started->setHtml(QString("<div>This is SPARTA-GUI version " SPARTA_GUI_VERSION "</div>") +
                     GETTING_STARTED);
    tabs->addTab(started, "Getting Started");

    auto *keys = new QTextBrowser(this);
    keys->setObjectName("shortcutlist");
    tabs->addTab(keys, "Keyboard Shortcuts");

    layout->addWidget(tabs, 1);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
}

void ShortcutsDialog::popup(Page page)
{
    // regenerate on every opening: the table must describe the menus as they
    // are now, not as they were when the dialog was first built
    auto *keys = qobject_cast<QTextBrowser *>(tabs->widget(Shortcuts));
    if (keys) keys->setHtml(shortcutsHtml());
    tabs->setCurrentIndex(int(page));
    show();
    raise();
    activateWindow();
}

QString ShortcutsDialog::shortcutsHtml() const
{
    QString html = QStringLiteral("<table cellspacing='0' cellpadding='3' width='100%'>");
    QString lastMenu;
    for (const auto &info : scanMenuBar(menubar)) {
        if (!info.action) continue;
        const QString key = info.action->shortcut().toString(QKeySequence::NativeText);
        if (key.isEmpty()) continue;
        const QString menu = info.path.section(QStringLiteral(" > "), 0, 0);
        if (menu != lastMenu) {
            html += QString("<tr><td colspan='2'><h3>%1</h3></td></tr>").arg(menu.toHtmlEscaped());
            lastMenu = menu;
        }
        html += QString("<tr><td><b>%1</b></td><td>%2</td></tr>")
                    .arg(key.toHtmlEscaped(), info.text.toHtmlEscaped());
    }
    // keys that are not menu actions and so cannot be harvested from the menus
    html += QStringLiteral(
        "<tr><td colspan='2'><h3>Editor</h3></td></tr>"
        "<tr><td><b>Ctrl+?</b></td><td>Documentation for the command on the current line"
        "</td></tr>"
        "<tr><td><b>Tab</b></td><td>Reformat the current line</td></tr>"
        "<tr><td><b>Shift+Tab</b></td><td>Show completions</td></tr>"
        "<tr><td><b>Ctrl+Home / Ctrl+End</b></td><td>Go to start / end of the deck</td></tr>"
        "<tr><td colspan='2'><h3>Other windows</h3></td></tr>"
        "<tr><td><b>Ctrl+W</b></td><td>Close the window</td></tr>"
        "</table>");
    return html;
}

// Local Variables:
// c-basic-offset: 4
// End:
