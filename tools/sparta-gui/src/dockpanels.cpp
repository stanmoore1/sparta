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
                                                              "dockVariables", "dockJobs",
                                                              "dockSweep"};
const char *const PANEL_TITLE[PanelManager::NPanels] = {"Output", "Charts", "Image",
                                                        "Slide Show", "Variables",
                                                        "Cluster Jobs", "Parameter Sweep"};

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
                emit panelOpened(i);
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
}

bool PanelManager::restoreLayout(QSettings &settings)
{
    const QByteArray blob = settings.value(Keys::DOCKSTATE).toByteArray();
    if (blob.isEmpty()) return false;
    return dm->restoreState(blob, Cfg::DOCK_LAYOUT_VERSION);
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
    // editor : charts/image column, horizontally 62:38
    splitArea(editorDock->dockAreaWidget(), 62);
    // editor row : output/variables, vertically 68:32
    splitArea(docks[Log]->dockAreaWidget(), 68);
    // charts : image, vertically within the right column 55:45
    splitArea(docks[Image]->dockAreaWidget(), 55);
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
    // on-demand tool panels (Cluster Jobs, Parameter Sweep) live tabbed with the
    // Output area but start hidden; they are shown from the menu when needed
    dm->addDockWidgetTabToArea(docks[Jobs], logArea);
    dm->addDockWidgetTabToArea(docks[Sweep], logArea);

    applySplitterProportions();

    for (int i = 0; i < NPanels; ++i)
        docks[i]->toggleView(false);
}

// Local Variables:
// c-basic-offset: 4
// End:
