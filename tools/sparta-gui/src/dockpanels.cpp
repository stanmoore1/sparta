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

#include "emptystate.h"

#include "constants.h"
#include "viewersource.h"

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
const char *const PANEL_OBJECT_NAME[PanelManager::NPanels] = {
    "dockOutput", "dockCharts", "dockViewer",     "dockVariables",   "dockSweep",
    "dockHistory", "dockDiagnostics", "dockProjectFiles"};
const char *const PANEL_TITLE[PanelManager::NPanels] = {
    "Output", "Charts", "Viewer",     "Variables",   "Parameter Sweep",
    "Run History", "Diagnostics", "Project Files"};

// What an empty panel says, and which action fills it. Installed as every
// panel's initial widget: a dock with nothing in it used to be a blank
// rectangle (and Qt-ADS refuses to lay out a dock with no widget at all, so
// workspaces could not even show it).
EmptyState *makePlaceholder(PanelManager::Panel panel)
{
    switch (panel) {
        case PanelManager::Log:
            return new EmptyState("No output yet",
                                  "Run the deck (Ctrl+Enter) and the console output lands here.");
        case PanelManager::Chart:
            return new EmptyState("No chart data yet",
                                  "The stats columns are charted here, live, during a run.");
        case PanelManager::Viewer:
            return new EmptyState(
                "Nothing rendered yet",
                "Create Image (Ctrl+I) renders the current state; frames from a dump image "
                "command and the 3D scene appear here as well.");
        case PanelManager::Variables:
            return new EmptyState("No variables yet",
                                  "Index variables from the deck and from Set Variables "
                                  "(Ctrl+Shift+V) are listed here during a run.");
        case PanelManager::Sweep:
            return new EmptyState("No sweep configured",
                                  "Tools > Parametric Sweep runs the deck repeatedly over "
                                  "index-variable ranges and tabulates the results.");
        case PanelManager::History:
            return new EmptyState("No runs archived yet",
                                  "Every finished run is archived and can be revisited here.");
        case PanelManager::Diagnostics:
            return new EmptyState("No problems found",
                                  "Check Input (Ctrl+K) lists its findings here; automatic "
                                  "checking marks them in the editor while you type.");
        case PanelManager::ProjectFiles:
            return new EmptyState("No input open",
                                  "The files beside the current deck are listed here.");
        default: return new EmptyState("Nothing here yet", QString());
    }
}

// Widgets that scroll themselves must not be wrapped in another scroll area.
// The Charts and Viewer hosts manage their own scrolling and zooming, and
// Output is a QPlainTextEdit, which is a scroll area already. Variables is a
// bare QLabel and does want the free wrapper.
CDockWidget::eInsertMode insertModeFor(PanelManager::Panel panel)
{
    switch (panel) {
        case PanelManager::Chart:
        case PanelManager::Viewer:
        case PanelManager::Log:
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
        docks[i]    = d;
        menuText[i] = PANEL_TITLE[i];

        // Qt-ADS retitles a dock's toggleViewAction() whenever the dock's own
        // title changes, and that action *is* the entry in the View menu.
        // Naming a panel after what it currently holds therefore rewrote the
        // menu: "Output Window" became "Output - in.circle - Run 1" the moment
        // a run started, and "Viewer Window" became the name of the last
        // snapshot. The menu stopped reading as a list of windows, and nothing
        // that looks an entry up by name -- someone scanning it, a screen
        // reader, the widget walker -- could find it again.
        //
        // Undone here rather than at each call site, so it holds however the
        // title comes to change. The dock's tab keeps the descriptive title,
        // which is where it is useful; the menu entry keeps its own name.
        connect(d, &CDockWidget::titleChanged, this, [this, i](const QString &) {
            QAction *entry = docks[i]->toggleViewAction();
            if (entry->text() != menuText[i]) entry->setText(menuText[i]);
        });
        d->setWidget(makePlaceholder(Panel(i)), insertModeFor(Panel(i)));

        connect(d, &CDockWidget::viewToggled, this, [this, i](bool open) {
            if (open) {
                restoreAreaVisibility(Panel(i));
                if (!applyingArrangement) {
                    // Six panels share the bottom dock area as tabs, so opening
                    // one can leave it behind whichever tab is already in front
                    // -- asking for a panel and apparently getting nothing.
                    // Not while a whole arrangement is being applied, where
                    // this would just leave whichever panel happened to be last.
                    docks[i]->setAsCurrentTab();
                    emit panelOpened(i);
                }
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

    if (old && keepOld && !EmptyState::isPlaceholder(old)) {
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
    setPanelTitle(panel, title);

    // A QDialog reacts to Escape by hiding itself (QDialog::reject ->
    // finished), which would otherwise leave the dock showing a blank content
    // pane. Map that back to "close the panel tab". The viewers used to be
    // dialogs and needed this; they are plain widgets now, which is the real
    // fix, but other panels may still be handed a dialog.
    if (auto *dialog = qobject_cast<QDialog *>(widget)) {
        connect(dialog, &QDialog::finished, this, [this, panel, dialog](int) {
            dialog->show();
            closePanel(panel);
        });
    }

    // A viewer source cannot close itself: QWidget::close() does nothing for a
    // widget that is not a window, so without this its own Ctrl+W and File >
    // Close would silently do nothing once it was docked.
    if (auto *source = qobject_cast<ViewerSource *>(widget)) {
        connect(source, &ViewerSource::closeRequested, this,
                [this, panel]() { closePanel(panel); });
        connect(source, &ViewerSource::titleChanged, this,
                [this, panel](const QString &name) { setPanelTitle(panel, name); });
    }
}

void PanelManager::setPanelMenuText(Panel panel, const QString &text)
{
    menuText[panel] = text;
    docks[panel]->toggleViewAction()->setText(text);
}

void PanelManager::setPanelTitle(Panel panel, const QString &title)
{
    // The View menu entry is preserved by the titleChanged connection made in
    // the constructor, so this is only the dock's own tab.
    docks[panel]->setWindowTitle(title);
}

void PanelManager::clearRunPanels()
{
    for (int i = 0; i < NPanels; ++i) {
        delete docks[i]->takeWidget();
        docks[i]->setWidget(makePlaceholder(Panel(i)), insertModeFor(Panel(i)));
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
    // Stored by name, not by index: the enum has been renumbered once already
    // (the Setup workspace was folded into Run), and a stored index then names
    // a different workspace -- or one past the end -- in the next version.
    settings.setValue(Keys::DOCKMODE, modeName(mode));
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
        // An index left by a version that still had the Setup workspace is not
        // a name, and is not worth decoding -- come up in Run, which is where
        // Setup's profiles belong now anyway.
        const QString m = settings.value(Keys::DOCKMODE).toString();
        for (int i = 0; i < NModes; ++i)
            if (modeName(Mode(i)) == m) mode = Mode(i);
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

void PanelManager::splitArea(CDockAreaWidget *area, int percent, Qt::Orientation orient)
{
    // Give @p area the requested share of the nearest enclosing splitter that
    // runs along @p orient.
    //
    // Addressing the splitter by orientation rather than by "the parent of the
    // area" matters because the same area sits in two of them: the editor is a
    // child of the vertical editor/output splitter *and*, one level up, of the
    // horizontal editor/right-column splitter. Keying off the immediate parent
    // silently retargets whichever one the dock tree happens to nest first, so
    // adding a panel could leave two proportions fighting over one splitter and
    // the other never set at all.
    //
    // Percentages must not be passed to setSizes() directly either: QSplitter
    // treats them as base sizes and hands the leftover space to the highest-
    // stretch child, which for us is always the ADS central widget (the editor)
    // -- so {62,38} would collapse a side panel to ~60px instead of ~38%.
    if (!area) return;

    QWidget *child = area;
    QSplitter *sp  = nullptr;
    for (QWidget *p = area->parentWidget(); p; child = p, p = p->parentWidget()) {
        auto *s = qobject_cast<QSplitter *>(p);
        if (s && s->orientation() == orient && s->count() > 1) {
            sp = s;
            break;
        }
    }
    if (!sp) return;

    const int idx = sp->indexOf(child);
    if (idx < 0) return;
    const int total = (orient == Qt::Horizontal) ? sp->width() : sp->height();
    if (total <= 0) return;

    // Hand @p area its share and divide the remainder among the other children
    // in their current proportions, so a third pane (the file navigator) keeps
    // the size it already had instead of being crushed to nothing.
    QList<int> sizes = sp->sizes();
    const int mine   = total * percent / 100;
    int othersNow    = 0;
    for (int i = 0; i < sizes.size(); ++i)
        if (i != idx) othersNow += sizes[i];

    const int rest = total - mine;
    for (int i = 0; i < sizes.size(); ++i) {
        if (i == idx) continue;
        sizes[i] = (othersNow > 0) ? rest * sizes[i] / othersNow : rest / (sizes.size() - 1);
    }
    sizes[idx] = mine;
    sp->setSizes(sizes);
}

void PanelManager::applySplitterProportions()
{
    // Only reserve space for a neighbour that actually has something open.
    // Forcing the editor to 62% when the whole right-hand column is closed --
    // as it is before a run has produced any output -- would leave 38% of the
    // window as an empty gap.
    auto anyOpen = [this](std::initializer_list<Panel> ps) {
        for (Panel p : ps)
            if (!docks[p]->isClosed()) return true;
        return false;
    };

    // project files navigator : everything else, horizontally 18:82 -- wide
    // enough for real file names rather than the sliver QSplitter would
    // otherwise give a freshly-opened left dock
    if (anyOpen({ProjectFiles}))
        splitArea(docks[ProjectFiles]->dockAreaWidget(), 18, Qt::Horizontal);
    // editor : right-hand column, horizontally. An even split while the column
    // is just the output, because a deck and its output deserve the same width;
    // less for the editor once the pictures are up, and least of all in
    // Visualize, which exists to give them the window.
    if (anyOpen({Log, Variables, Sweep, History, Diagnostics, Chart, Viewer})) {
        int editorShare = 50;
        if (mode == Visualize)
            editorShare = 25;
        else if (anyOpen({Chart, Viewer}))
            editorShare = 40;
        splitArea(editorDock->dockAreaWidget(), editorShare, Qt::Horizontal);
    }
    // charts : image, vertically within the right column. Split it evenly: the
    // viewer scales its render to whatever room it has and still reads
    // correctly, whereas the chart spends a fixed ~75px on its two control rows
    // before the plot gets anything, so an uneven split costs the chart far
    // more than it gains the image.
    // The rules below only make sense once the user has dragged panels apart
    // into their own areas; in the default tabbed layout they share one area
    // and there is nothing to apportion.
    if (anyOpen({Chart}) && anyOpen({Viewer}) &&
        docks[Chart]->dockAreaWidget() != docks[Viewer]->dockAreaWidget())
        splitArea(docks[Viewer]->dockAreaWidget(), 50, Qt::Vertical);
    // and the output's share of that column when it is up alongside them
    if (anyOpen({Log}) && anyOpen({Chart, Viewer}) &&
        docks[Log]->dockAreaWidget() != docks[Chart]->dockAreaWidget() &&
        docks[Log]->dockAreaWidget() != docks[Viewer]->dockAreaWidget())
        splitArea(docks[Log]->dockAreaWidget(), 34, Qt::Vertical);
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
    // One right-hand area beside the editor, holding every panel as a TAB
    // rather than a vertical stack. Splitting was how the window starved:
    // with three panels open, each pane -- and the editor -- was too small to
    // read. Tabs give whichever panel is in front the whole column, switching
    // is one click (or the View menu), and anything the user *wants* side by
    // side can still be dragged out into a split, which the workspace then
    // remembers in its perspective.
    CDockAreaWidget *logArea = dm->addDockWidget(ads::RightDockWidgetArea, docks[Log]);
    dm->addDockWidgetTabToArea(docks[Chart], logArea);
    dm->addDockWidgetTabToArea(docks[Viewer], logArea);
    dm->addDockWidgetTabToArea(docks[Variables], logArea);
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
    // Run: writing a deck and watching it run -- the editor and its output side
    // by side, plus the variables tabbed behind the output. This is also where
    // the deck is prepared: a separate Setup workspace showed the same two
    // columns and was not worth the click. The linter's findings and the file
    // navigator are a keystroke away in the View menu when they are wanted.
    {PanelManager::Log, PanelManager::Variables},
    // Analyze: the plots, with the window given over to them. A run started
    // from here raises the chart and nothing else.
    {PanelManager::Chart},
    // Visualize: the pictures, with the window given over to them
    {PanelManager::Viewer},
};
} // namespace

bool PanelManager::modeShows(Mode mode, Panel panel)
{
    if (mode < 0 || mode >= NModes) return false;
    return MODE_PANELS[mode].contains(panel);
}

QString PanelManager::modeName(Mode mode)
{
    switch (mode) {
        case RunMode: return QStringLiteral("Run");
        case Analyze: return QStringLiteral("Analyze");
        case Visualize: return QStringLiteral("Visualize");
        default: return QString();
    }
}

QString PanelManager::panelName(Panel panel)
{
    if (panel < 0 || panel >= NPanels) return {};
    return QString::fromLatin1(PANEL_TITLE[panel]);
}

QString PanelManager::perspectiveName(Mode mode)
{
    return QStringLiteral("mode.") + modeName(mode);
}

void PanelManager::applyModeDefault(Mode m)
{
    const QList<Panel> &want = MODE_PANELS[m];
    for (int i = 0; i < NPanels; ++i) {
        // Every dock always holds a widget -- real content or its EmptyState
        // card -- so a mode can simply show its panels. Before the cards, a
        // dock with no widget could not be shown at all (Qt-ADS never lays out
        // an empty dock area), and a workspace entered before any run was a
        // bare editor with no explanation.
        docks[i]->toggleView(want.contains(Panel(i)));
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
