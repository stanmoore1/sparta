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

#ifndef DOCKPANELS_H
#define DOCKPANELS_H

// Single-window docked layout for the Output/Charts/Image/Slide Show/Variables
// views, built on the vendored Qt-ADS (Advanced Docking System). All ADS-facing
// logic is confined here; spartagui.{h,cpp} only calls the methods below, so
// the secondary-window classes (LogWindow, ChartWindow, ImageViewer, SlideShow)
// stay untouched for easier upstream (LAMMPS-GUI) cherry-picking.

#include <QList>
#include <QObject>
#include <QPointer>

class QAction;
class QMainWindow;
class QSettings;
class QWidget;

namespace ads {
class CDockAreaWidget;
class CDockManager;
class CDockWidget;
} // namespace ads

/**
 * @brief Owns the Qt-ADS dock manager and the five secondary-view dock panels
 *
 * PanelManager creates a ads::CDockManager on top of the main window, installs
 * the editor as the (non-closable) central widget, and creates one stable,
 * initially-hidden ads::CDockWidget per secondary view (Output, Charts, Image,
 * Slide Show, Variables) in the default layout. Call sites hand it the actual
 * view widget (LogWindow, ChartWindow, ...) via @ref setPanelWidget instead of
 * showing/hiding a top-level window.
 */
class PanelManager : public QObject {
    Q_OBJECT

public:
    /** @brief One dock panel slot; also indexes the internal dock/action arrays */
    enum Panel {
        Log, Chart, Image, Slide, Variables, Sweep, History, Diagnostics, ProjectFiles,
        NPanels
    };

    /**
     * @brief Task-oriented workspace mode
     *
     * Showing every panel at once leaves too little room for any of them, so
     * the panels are grouped by what the user is actually doing: preparing a
     * deck, watching a run, or studying its results. Each mode is backed by a
     * Qt-ADS perspective, so rearranging panels inside a mode is remembered
     * per mode rather than globally.
     */
    enum Mode { Setup, RunMode, Analyze, NModes };

    /**
     * @brief Build the dock manager, central editor dock, and the five stable panels
     * @param mainWindow Main window the dock manager installs itself into
     * @param editor     Editor widget to install as the central (non-closable) dock
     */
    PanelManager(QMainWindow *mainWindow, QWidget *editor);
    ~PanelManager() override;
    PanelManager()                                  = delete;
    PanelManager(const PanelManager &)               = delete;
    PanelManager(PanelManager &&)                    = delete;
    PanelManager &operator=(const PanelManager &)    = delete;
    PanelManager &operator=(PanelManager &&)         = delete;

    /** @brief The stable dock widget for a panel (always valid, never re-created) */
    ads::CDockWidget *dock(Panel panel) const;

    /** @brief Checkable action toggling a panel's visibility, for the View menu */
    QAction *toggleViewAction(Panel panel) const;

    /**
     * @brief Install a view widget into a panel's stable dock
     * @param panel   Panel slot to update
     * @param widget  View widget to take ownership of (e.g. a LogWindow)
     * @param title   Window/tab title to apply to the stable dock
     * @param keepOld If true, the panel's previous widget is moved into a new
     *                closable, self-deleting archived dock tabbed alongside
     *                the panel instead of being deleted (mirrors the
     *                "replace on new run" preferences being off)
     *
     * Always detaches the dock's current widget first (Qt-ADS discards it
     * otherwise): safe to call even when @p panel currently holds @p widget's
     * predecessor from an earlier run.
     */
    void setPanelWidget(Panel panel, QWidget *widget, const QString &title, bool keepOld = false);

    /**
     * @brief Tear down all run-owned panel content before a new document/run
     *
     * Detaches and deletes each panel's current inner widget, deletes any
     * archived (kept-old) docks, and hides the stable docks -- the layout
     * itself (splitter geometry, tab groupings) survives. Callers still null
     * out their own view widget pointers afterward.
     */
    void clearRunPanels();

    /** @brief Show a panel's dock (creating no content -- see setPanelWidget) */
    void openPanel(Panel panel);
    /** @brief Hide a panel's dock */
    void closePanel(Panel panel);
    /** @brief True if the panel's dock is currently open (visible) */
    bool isPanelOpen(Panel panel) const;

    /** @brief Persist the current dock layout (geometry, tabs, open/closed state) */
    void saveLayout(QSettings &settings) const;
    /** @brief Restore a previously saved layout; returns false if none/incompatible
     *  (default layout is left in place on failure) */
    bool restoreLayout(QSettings &settings);
    /** @brief Reset the dock layout to the initial default arrangement */
    void applyDefaultLayout();

    /**
     * @brief Switch to a workspace mode
     *
     * Stashes the current mode's arrangement as a perspective, then restores
     * @p mode's previously-stored arrangement, or builds it from the mode's
     * default panel set the first time it is entered. Only dock visibility
     * changes: panel *content* is never destroyed, so a run's output survives
     * a round trip through the other modes.
     */
    void applyMode(Mode mode);

    /** @brief The workspace mode currently displayed */
    Mode currentMode() const { return mode; }

    /** @brief Discard the current mode's stored arrangement and rebuild it
     *  from that mode's default panel set (other modes are left alone) */
    void resetCurrentMode();

    /** @brief Human-readable name of a mode, for menus and the mode switch */
    static QString modeName(Mode mode);

signals:
    /** @brief Emitted when a panel is opened by user action (for lazy view creation) */
    void panelOpened(int panel);

    /** @brief Emitted after the displayed workspace mode changes */
    void modeChanged(int mode);

private:
    /** @brief (Re-)apply the default splitter proportions between the editor and
     *  the Charts/Image and Output/Variables areas. */
    void applySplitterProportions();

    /** @brief Size the two-child splitter containing @p area so its first child
     *  gets @p firstPercent of the current extent (see the .cpp for why explicit
     *  pixel sizes are required rather than percentages). */
    void splitArea(ads::CDockAreaWidget *area, int firstPercent);

    /** @brief Un-hide a panel's dock area and restore proportions after it is
     *  reopened. Deferred to the next event loop -- see the implementation for
     *  the Qt-ADS quirk this works around. */
    void restoreAreaVisibility(Panel panel);

    /** @brief Show exactly the panels in @p mode's default set, hide the rest */
    void applyModeDefault(Mode mode);

    /** @brief Stash the current arrangement as the current mode's perspective */
    void captureCurrentMode();

    /** @brief Qt-ADS perspective name backing a mode */
    static QString perspectiveName(Mode mode);

    ads::CDockManager *dm;
    ads::CDockWidget *editorDock;
    ads::CDockWidget *docks[NPanels];
    QList<QPointer<ads::CDockWidget>> archived;
    int archiveSeq = 0;
    Mode mode = Setup;   ///< fresh profiles start where the work starts: the deck

    /** @brief Whether a mode's arrangement has ever been established.
     *  A mode that was never entered has no arrangement worth keeping, and
     *  stashing the startup state (every dock closed) under its name would
     *  make the mode come up empty the first time it is selected. */
    bool modeEstablished[NModes] = {false, false, false};

    /** @brief Suppresses panelOpened while a whole arrangement is being applied.
     *  Switching modes must not trigger the lazy view creation that opening a
     *  panel by hand does -- entering Analyze would otherwise try to render a
     *  snapshot with no simulation box loaded. */
    bool applyingArrangement = false;
};

#endif // DOCKPANELS_H

// Local Variables:
// c-basic-offset: 4
// End:
