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

#include "dockpanels.h"

#include "constants.h"

#include "DockAreaWidget.h"
#include "DockManager.h"
#include "DockWidget.h"

#include <QAction>
#include <QDialog>
#include <QMainWindow>
#include <QSettings>
#include <QSplitter>
#include <QTimer>
#include <QWidget>

#include <utility>

using ads::CDockAreaWidget;
using ads::CDockManager;
using ads::CDockWidget;

namespace {

// stable, unique object names -- required by CDockManager::saveState/restoreState
const char *const PANEL_OBJECT_NAME[PanelManager::NPanels] = {"dockOutput", "dockCharts",
                                                              "dockImage", "dockSlideShow",
                                                              "dockVariables", "dockSweep",
                                                              "dockHistory", "dockDiagnostics",
                                                              "dockProjectFiles"};
const char *const PANEL_TITLE[PanelManager::NPanels] = {"Output", "Charts", "Image",
                                                        "Slide Show", "Variables",
                                                        "Parameter Sweep", "Run History",
                                                        "Diagnostics", "Project Files"};

// Chart/Image/Slide Show host widgets manage their own scrolling/zooming and
// must not be wrapped in an extra QScrollArea; Output (QPlainTextEdit) and
// Variables (a bare QLabel) get the default auto-detected behavior -- the
// QLabel in particular gets a free scroll wrapper it would not otherwise have.
CDockWidget::eInsertMode insertModeFor(PanelManager::Panel panel)
{
    switch (panel) {
        case PanelManager::Chart:
        case PanelManager::Image:
        case PanelManager::Slide:
            return CDockWidget::ForceNoScrollArea;
        default:
            return CDockWidget::AutoScrollArea;
    }
}

} // namespace

PanelManager::PanelManager(QMainWindow *mainWindow, QWidget *editor) : QObject(mainWindow)
{
    dm = new CDockManager(mainWindow);

    editorDock = new CDockWidget(dm, "Editor");
    editorDock->setObjectName("dockEditor");
    editorDock->setWidget(editor, CDockWidget::ForceNoScrollArea);
    dm->setCentralWidget(editorDock);

    for (int i = 0; i < NPanels; ++i) {
        auto *d = new CDockWidget(dm, PANEL_TITLE[i]);
        d->setObjectName(PANEL_OBJECT_NAME[i]);
        docks[i] = d;
        connect(d, &CDockWidget::viewToggled, this, [this, i](bool open) {
            if (open) {
                restoreAreaVisibility(Panel(i));
                if (!applyingArrangement) emit panelOpened(i);
            }
        });
    }

    applyDefaultLayout();
}

PanelManager::~PanelManager() = default;

ads::CDockWidget *PanelManager::dock(Panel panel) const
{
    return docks[panel];
}

QAction *PanelManager::toggleViewAction(Panel panel) const
{
    return docks[panel]->toggleViewAction();
}

void PanelManager::setPanelWidget(Panel panel, QWidget *widget, const QString &title, bool keepOld)
{
    CDockWidget *d = docks[panel];
    // Qt-ADS' setWidget() silently discards whatever widget is already
    // installed, so it must always be detached first regardless of keepOld.
    QWidget *old = d->takeWidget();

    if (old && keepOld) {
        auto *arch = new CDockWidget(dm, d->windowTitle());
        arch->setObjectName(QString("%1Archived%2").arg(PANEL_OBJECT_NAME[panel]).arg(++archiveSeq));
        arch->setFeature(CDockWidget::DockWidgetDeleteOnClose, true);
        arch->setFeature(CDockWidget::DeleteContentOnClose, true);
        arch->setWidget(old, insertModeFor(panel));
        dm->addDockWidgetTabToArea(arch, d->dockAreaWidget());
        archived.append(arch);
    } else {
        delete old;
    }

    d->setWidget(widget, insertModeFor(panel));
    d->setWindowTitle(title);

    // A QDialog (ImageViewer, SlideShow) reacts to Escape by hiding itself
    // (QDialog::reject -> finished), which would otherwise leave the dock
    // showing a blank content pane. Map that back to "close the panel tab".
    if (auto *dialog = qobject_cast<QDialog *>(widget)) {
        connect(dialog, &QDialog::finished, this, [this, panel, dialog](int) {
            dialog->show();
            closePanel(panel);
        });
    }
}

void PanelManager::clearRunPanels()
{
    for (int i = 0; i < NPanels; ++i) {
        delete docks[i]->takeWidget();
        docks[i]->toggleView(false);
    }
    for (const auto &a : std::as_const(archived))
        delete a;
    archived.clear();
}

void PanelManager::openPanel(Panel panel)
{
    docks[panel]->toggleView(true);
}

void PanelManager::closePanel(Panel panel)
{
    docks[panel]->toggleView(false);
}

bool PanelManager::isPanelOpen(Panel panel) const
{
    return !docks[panel]->isClosed();
}

void PanelManager::saveLayout(QSettings &settings) const
{
    settings.setValue(Keys::DOCKSTATE, dm->saveState(Cfg::DOCK_LAYOUT_VERSION));
    // stash how the user left the mode they are currently in, then persist all
    // of the per-mode arrangements together with the mode to reopen on
    settings.setValue(Keys::DOCKMODE, int(mode));
    settings.setValue(Keys::PERSPECTIVE_VERSION, Cfg::DOCK_LAYOUT_VERSION);
    const_cast<PanelManager *>(this)->captureCurrentMode();
    dm->savePerspectives(settings);
}

bool PanelManager::restoreLayout(QSettings &settings)
{
    // Perspectives are stored by Qt-ADS without a version stamp, so guard them
    // with our own: a blob written before a panel was added or removed names
    // docks that no longer exist and must be discarded rather than applied.
    if (settings.value(Keys::PERSPECTIVE_VERSION, 0).toInt() == Cfg::DOCK_LAYOUT_VERSION) {
        dm->loadPerspectives(settings);
        // a mode with a stored perspective was configured in an earlier session
        const QStringList have = dm->perspectiveNames();
        for (int i = 0; i < NModes; ++i)
            modeEstablished[i] = have.contains(perspectiveName(Mode(i)));
        const int m = settings.value(Keys::DOCKMODE, int(Setup)).toInt();
        if (m >= 0 && m < NModes) mode = Mode(m);
    } else {
        settings.remove(QStringLiteral("Perspectives"));
    }

    applyingArrangement = true;
    const QByteArray blob = settings.value(Keys::DOCKSTATE).toByteArray();
    bool ok = false;
    if (!blob.isEmpty()) ok = dm->restoreState(blob, Cfg::DOCK_LAYOUT_VERSION);
    // with no usable saved session, come up in the current mode's default set
    if (!ok) applyModeDefault(mode);
    modeEstablished[mode] = true;
    applyingArrangement   = false;
    return ok;
}

void PanelManager::splitArea(CDockAreaWidget *area, int firstPercent)
{
    // Give the two-child splitter that contains @p area explicit pixel sizes,
    // computed from its current width/height so they sum to the actual extent.
    // Percentages must not be passed to setSplitterSizes() directly: QSplitter
    // treats them as base sizes and hands the leftover space to the highest-
    // stretch child, which for us is always the ADS central widget (the editor)
    // -- so {62,38} would collapse a side panel to ~60px instead of ~38%.
    if (!area) return;
    auto *sp = qobject_cast<QSplitter *>(area->parentWidget());
    if (!sp || sp->count() != 2) return;
    const int total = (sp->orientation() == Qt::Horizontal) ? sp->width() : sp->height();
    if (total <= 0) return;
    const int first = total * firstPercent / 100;
    dm->setSplitterSizes(area, {first, total - first});
}

void PanelManager::applySplitterProportions()
{
    // Only reserve space for a neighbour that actually has something open.
    // Forcing the editor to 62% when the whole right-hand column is closed --
    // as it is in the Setup workspace -- would leave 38% of the window as an
    // empty gap.
    auto anyOpen = [this](std::initializer_list<Panel> ps) {
        for (Panel p : ps)
            if (!docks[p]->isClosed()) return true;
        return false;
    };

    // project files navigator : everything else, horizontally 18:82 -- wide
    // enough for real file names rather than the sliver QSplitter would
    // otherwise give a freshly-opened left dock
    if (anyOpen({ProjectFiles})) splitArea(docks[ProjectFiles]->dockAreaWidget(), 18);
    // editor : charts/image column, horizontally 62:38
    if (anyOpen({Chart, Image, Slide})) splitArea(editorDock->dockAreaWidget(), 62);
    // editor row : output/variables/tools, vertically 68:32
    if (anyOpen({Log, Variables, Sweep, History, Diagnostics}))
        splitArea(docks[Log]->dockAreaWidget(), 68);
    // charts : image, vertically within the right column 55:45
    if (anyOpen({Chart}) && anyOpen({Image, Slide})) splitArea(docks[Image]->dockAreaWidget(), 55);
}

void PanelManager::restoreAreaVisibility(Panel panel)
{
    // When every dock in an area has been closed (the startup/reset state),
    // Qt-ADS hides the CDockAreaWidget and does not un-hide it when one of its
    // docks is reopened -- leaving a visible-but-zero-height pane. viewToggled
    // also fires synchronously, before Qt has processed the re-show, so this is
    // deferred to the next event-loop turn: show the area explicitly, then
    // restore the splitter proportions (which QSplitter ignores while the child
    // is hidden).
    QTimer::singleShot(0, this, [this, panel]() {
        if (docks[panel]->isClosed()) return;
        if (auto *area = docks[panel]->dockAreaWidget()) area->setVisible(true);
        applySplitterProportions();
    });
}

void PanelManager::applyDefaultLayout()
{
    CDockAreaWidget *chartArea = dm->addDockWidget(ads::RightDockWidgetArea, docks[Chart]);
    CDockAreaWidget *imageArea =
        dm->addDockWidget(ads::BottomDockWidgetArea, docks[Image], chartArea);
    dm->addDockWidgetTabToArea(docks[Slide], imageArea);
    CDockAreaWidget *logArea = dm->addDockWidget(ads::BottomDockWidgetArea, docks[Log]);
    dm->addDockWidgetTabToArea(docks[Variables], logArea);
    // on-demand tool panels (Parameter Sweep, Run History) live tabbed with the
    // Output area but start hidden; they are shown from the menu when needed
    dm->addDockWidgetTabToArea(docks[Sweep], logArea);
    dm->addDockWidgetTabToArea(docks[History], logArea);
    dm->addDockWidgetTabToArea(docks[Diagnostics], logArea);
    // Project Files is a navigator; dock it on the left of the editor
    dm->addDockWidget(ads::LeftDockWidgetArea, docks[ProjectFiles]);

    applySplitterProportions();

    for (int i = 0; i < NPanels; ++i)
        docks[i]->toggleView(false);
}

// ---------------------------------------------------------------------------
// Workspace modes
// ---------------------------------------------------------------------------

// Panels shown when a mode is entered for the first time. Deliberately small
// sets: the whole point of modes is that the window is not carrying panels the
// user is not currently looking at. Anything omitted here is still one click
// away in the View menu, and opening it by hand becomes part of that mode's
// remembered arrangement.
namespace {
const QList<PanelManager::Panel> MODE_PANELS[PanelManager::NModes] = {
    // Setup: writing the deck -- the file navigator and the linter's findings
    {PanelManager::ProjectFiles, PanelManager::Diagnostics},
    // Run: watching a run -- console output, live plots, and the variables
    {PanelManager::Log, PanelManager::Variables, PanelManager::Chart},
    // Analyze: studying results -- plots and rendered snapshots, with the log
    // kept for reference
    {PanelManager::Chart, PanelManager::Image, PanelManager::Slide, PanelManager::Log},
};
} // namespace

QString PanelManager::modeName(Mode mode)
{
    switch (mode) {
        case Setup: return QStringLiteral("Setup");
        case RunMode: return QStringLiteral("Run");
        case Analyze: return QStringLiteral("Analyze");
        default: return QString();
    }
}

QString PanelManager::perspectiveName(Mode mode)
{
    return QStringLiteral("mode.") + modeName(mode);
}

void PanelManager::applyModeDefault(Mode m)
{
    const QList<Panel> &want = MODE_PANELS[m];
    for (int i = 0; i < NPanels; ++i) {
        // A panel belonging to this mode is only shown once it has something to
        // show. Opening a dock that holds no widget yet gives Qt-ADS an empty
        // dock area it never lays out properly -- the panel then stays invisible
        // even after content arrives. Panels fill in as work produces them (a
        // run creating the charts, the linter creating diagnostics), and
        // setPanelWidget() opens them at that point if the mode calls for it.
        const bool wanted = want.contains(Panel(i));
        docks[i]->toggleView(wanted && docks[i]->widget() != nullptr);
    }
    applySplitterProportions();
}

void PanelManager::captureCurrentMode()
{
    // Only a mode that has actually been shown has an arrangement worth
    // keeping. Capturing one that was never entered would store the startup
    // state -- every dock closed -- and that mode would then come up empty
    // the first time the user selects it.
    if (!modeEstablished[mode]) return;
    dm->addPerspective(perspectiveName(mode));
}

void PanelManager::applyMode(Mode m)
{
    if (m < 0 || m >= NModes) return;

    // remember how the user left the mode being departed
    if (!applyingArrangement) captureCurrentMode();

    mode = m;

    // Applying a whole arrangement re-opens docks wholesale. That must not be
    // mistaken for the user opening a panel by hand, which triggers lazy view
    // creation -- so hold panelOpened down for the duration.
    applyingArrangement = true;
    if (modeEstablished[m] && dm->perspectiveNames().contains(perspectiveName(m)))
        dm->openPerspective(perspectiveName(m));
    else
        applyModeDefault(m);
    modeEstablished[m]  = true;
    applyingArrangement = false;

    // the ADS quirk worked around in restoreAreaVisibility() also applies to a
    // freshly restored arrangement, so re-assert proportions once Qt has caught up
    QTimer::singleShot(0, this, [this]() { applySplitterProportions(); });

    emit modeChanged(int(mode));
}

void PanelManager::resetCurrentMode()
{
    dm->removePerspective(perspectiveName(mode));
    applyingArrangement = true;
    applyModeDefault(mode);
    applyingArrangement = false;
    QTimer::singleShot(0, this, [this]() { applySplitterProportions(); });
}

// Local Variables:
// c-basic-offset: 4
// End:
