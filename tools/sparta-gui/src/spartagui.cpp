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

#include "spartagui.h"

#include "aboutdialog.h"
#include "actionmetadata.h"
#include "commandpalette.h"
#include "chartviewer.h"
#include "codeeditor.h"
#include "shortcutsdialog.h"
#include "dockpanels.h"
#include "welcomescreen.h"
#include "fileviewer.h"
#include "findandreplace.h"
#include "helpers.h"
#include "highlighter.h"
#include "imageviewer.h"
#include "spartarunner.h"
#include "logwindow.h"
#include "plotdata.h"
#include "paraviewdialog.h"
#include "plotdatadialog.h"
#include "preferences.h"
#include "libraryacquire.h"
#include "setupcard.h"
#include "runhistory.h"
#include "snippets.h"
#include "stlimportwizard.h"
#include "surfreportdialog.h"
#include "sweeppanel.h"
#include "setvariables.h"
#include "slideshow.h"
#include "viewerpanel.h"
#include "viewerwindow.h"
#include "stdcapture.h"
#include "urldownloader.h"
#if defined(SPARTA_GUI_HAVE_VTK)
#include "vtkscene.h"
#endif

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QSplitter>
#include <QVBoxLayout>
#include <QByteArray>
#include <QCheckBox>
#include <QClipboard>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QSysInfo>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFontInfo>
#include <QGridLayout>
#include <QGuiApplication>
#include <QKeySequence>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QStandardPaths>
#include <QStackedWidget>
#include <QStatusBar>
#include <QStringList>
#include <QTextStream>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QUrl>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

#include "constants.h"

namespace {

// read one thermo column value, converting from its native datatype
double lastThermoData(SpartaWrapper &sparta, int datatype, int column)
{
    if (datatype == 0) // int
        return sparta.lastThermoAs<int>("data", column);
    if (datatype == 2) // double
        return sparta.lastThermoAs<double>("data", column);
    if (datatype == 4) // bigint
        return static_cast<double>(sparta.lastThermoAs<int64_t>("data", column));
    return 0.0;
}

// export the https_proxy environment variable into the SPARTA instance, taken
// from the environment or, failing that, from the preferences
void applyProxySetting(SpartaWrapper &sparta, QSettings &settings)
{
    auto proxy = QString::fromLocal8Bit(qgetenv("https_proxy"));
    if (proxy.isEmpty()) proxy = settings.value(Keys::HTTPS_PROXY, "").toString();
    if (!proxy.isEmpty()) sparta.command(QString("shell putenv https_proxy=") + proxy);
}

// shown as the initial editor content and appended to the About info;
// intentionally empty for SPARTA-GUI (no citation banner)
const QString citeme;
} // namespace

void SpartaGui::setupUi(QSettings &settings, QFont &allFont, QFont &monoFont)
{
    setObjectName("SpartaGui");
    setWindowTitle("SPARTA-GUI");
    setWindowIcon(QIcon(Cfg::MAIN_ICON));

    // set up central widget
    textEdit = new CodeEditor(this);
    textEdit->setEnabled(true);
    textEdit->setAcceptDrops(true);
    // the editor applies its own banner watermark stylesheet in its constructor;
    // the color scheme (background/foreground) is applied later via applyEditorColorScheme()
    textEdit->setMinimumSize(Cfg::EDITOR_MIN_WIDTH, Cfg::EDITOR_MIN_HEIGHT);

    // the central area shows either the welcome screen (landing) or the editor;
    // a QStackedWidget lets them swap without touching the docked layout
    welcome      = new WelcomeScreen(this);
    centralStack = new QStackedWidget(this);
    centralStack->addWidget(welcome);
    centralStack->addWidget(textEdit);
    centralStack->setCurrentWidget(textEdit);
    connect(welcome, &WelcomeScreen::newFileRequested, this, &SpartaGui::newDocument);
    connect(welcome, &WelcomeScreen::browseRequested, this, &SpartaGui::open);
    connect(welcome, &WelcomeScreen::openFileRequested, this,
            [this](const QString &path) { openFile(path); });
    connect(welcome, &WelcomeScreen::openExampleRequested, this, &SpartaGui::openExamplePath);
    // the gallery is curated highlights; the full example list is this menu
    connect(welcome, &WelcomeScreen::browseExamplesRequested, this,
            [this]() { exampleMenu->popup(QCursor::pos()); });
    connect(welcome, &WelcomeScreen::helpRequested, this, &SpartaGui::help);
    connect(welcome, &WelcomeScreen::docsRequested, this, &SpartaGui::howto);

    // The setup card sits above the central stack rather than on the welcome
    // screen, so it is there whether the session opens on the welcome page or
    // straight into a file -- a user who launched with a deck on the command
    // line has exactly as much need to be told the simulator is missing.
    setupCard = new SetupCard(this);
    setupCard->hide();
    connect(setupCard, &SetupCard::downloadRequested, this, &SpartaGui::downloadLibrary);
    connect(setupCard, &SetupCard::browseRequested, this, &SpartaGui::browseForLibrary);
    connect(setupCard, &SetupCard::helpRequested, this, &SpartaGui::howto);

    auto *centralArea = new QWidget(this);
    auto *centralBox  = new QVBoxLayout(centralArea);
    centralBox->setContentsMargins(0, 0, 0, 0);
    centralBox->setSpacing(0);
    centralBox->addWidget(setupCard);
    centralBox->addWidget(centralStack, 1);

    // docked panel layout (Output/Charts/Image/Slide Show/Variables) replaces
    // setCentralWidget(): PanelManager installs the central area as the
    // (non-closable) central dock itself
    panels = new PanelManager(this, centralArea);

    // set up menu bar and menus with their actions and shortcuts
    menubar = new QMenuBar(this);
    createFileMenu();
    createEditMenu();
    createRunMenu();
    createToolsMenu();
    createViewMenu();
    createAboutMenu();
    setMenuBar(menubar);

    // Status bar
    createStatusBar();
    // and the toolbar above it, which shares the menus' QActions
    createToolBar();

    // Keep the View-menu entries and the status-bar switch in step with the
    // active mode, and give the mode the panels it expects to find populated.
    // Panels holding run output (Output, Charts, Image, Slide Show) are left
    // alone: they fill up when a run produces something, and creating them
    // here would render a snapshot with no simulation box loaded.
    connect(panels, &PanelManager::modeChanged, this, [this](int mode) {
        syncModeControls(mode);
        if (!startupComplete) return;
        // The panels are opened explicitly after their content exists: a mode
        // only shows panels that already have a widget, so a panel created
        // here would otherwise stay hidden until the next mode switch.
        //
        // Setup deliberately has no case here. It is the editing screen -- the
        // deck on the left and its output on the right, splitting the width
        // evenly, and nothing else. Opening the linter and the file navigator
        // here as well left the editor squeezed into a middle column between
        // them, which is the opposite of what the mode is for; both are one
        // entry away in the View menu, and the panelOpened handler below
        // creates them when they are asked for.
        switch (mode) {
            case PanelManager::RunMode:
                if (!varwindow) createVariableWindow();
                panels->openPanel(PanelManager::Variables);
                break;
            case PanelManager::Analyze:
                // The plots, with the window given over to them. The chart panel
                // is built lazily by a run, so before one has happened this mode
                // showed the deck and nothing else.
                ensureChartPanel();
                panels->openPanel(PanelManager::Chart);
                break;
            case PanelManager::Visualize:
                ensureViewerPanel();
                panels->openPanel(PanelManager::Viewer);
#if defined(SPARTA_GUI_HAVE_VTK)
                // The 3D view came up empty here, with its own Filters menu and
                // no other tab beside it to go back to. Entering the workspace
                // that exists to show it is as clear a request for its content
                // as there is, so fill it: the box and whatever surfaces the
                // deck reads before a run, everything after one.
                refreshDocked3DScene();
#endif
                break;
            default: break;
        }
    });
    syncModeControls(int(panels->currentMode()));

    // document settings
    auto *document = textEdit->document();
    document->setPlainText(citeme);
    document->setModified(false);
    highlighter = new Highlighter(document);
    connect(document, &QTextDocument::modificationChanged, this, &SpartaGui::modified);

    // auto-lint: re-validate the deck a short moment after the cursor lands on a
    // new line (covers pressing Enter and moving the cursor between lines).  The
    // debounce timer coalesces rapid movements into a single check; the feature
    // can be turned off in Preferences (autoLintEnabled) but has no menu entry.
    autoLintTimer = new QTimer(this);
    autoLintTimer->setSingleShot(true);
    autoLintTimer->setInterval(500);
    connect(autoLintTimer, &QTimer::timeout, this, &SpartaGui::autoCheckInput);
    connect(textEdit, &QPlainTextEdit::cursorPositionChanged, this, [this]() {
        if (!autoLintEnabled) return;
        const int block = textEdit->textCursor().blockNumber();
        if (block == lastLintBlock) return; // same line: nothing new to lint
        lastLintBlock = block;
        autoLintTimer->start();
    });

    // apply font settings
    setFont(allFont);
    textEdit->setFont(monoFont);
    document->setDefaultFont(monoFont);

    // apply the stored editor color scheme (token colors + editor background)
    applyEditorColorScheme();

    // set width and height of main window
    // use default so the background logo is fully shown
    // use last values unless overridden from command-line
    // do not accept a geometry smaller than minimum, revert to default instead
    if (mainx < Cfg::MINIMUM_WIDTH)
        mainx = settings.value(Keys::MAINX, Cfg::DEFAULT_MAIN_WIDTH).toInt();
    if (mainy < Cfg::MINIMUM_HEIGHT)
        mainy = settings.value(Keys::MAINY, Cfg::DEFAULT_MAIN_HEIGHT).toInt();
    resize(mainx, mainy);

    createVariableWindow();
}

template <typename Func>
QAction *SpartaGui::addMenuAction(QMenu *menu, const QString &iconpath, const QString &text,
                                  const QString &shortcut, Func slot)
{
    auto *action = new QAction(iconpath.isEmpty() ? QIcon() : QIcon(iconpath), text, this);
    if (!shortcut.isEmpty()) action->setShortcut(QKeySequence(shortcut));
    connect(action, &QAction::triggered, this, slot);
    menu->addAction(action);
    return action;
}

void SpartaGui::createFileMenu()
{
    auto *menu = menubar->addMenu("&File");
    // Alt+Home, not Ctrl+Home: Ctrl+Home is cursor-to-start-of-document in
    // every text editor including ours, and stealing it from the editor to
    // switch pages would be the kind of surprise this menu exists to avoid.
    addMenuAction(menu, ":/icons/help-faq.svg", "&Welcome Screen", "Alt+Home",
                  [this]() { showWelcome(); });
    menu->addSeparator();
    newAction = addMenuAction(menu, ":/icons/document-new.svg", "&New Input File", "Ctrl+N",
                              &SpartaGui::newDocument);
    openAction = addMenuAction(menu, ":/icons/document-open.svg", "&Open Input File", "Ctrl+O",
                               &SpartaGui::open);
    exampleMenu = menu->addMenu(QIcon(":/icons/document-open.svg"), "Open &Example");
    exampleMenu->setEnabled(false);
    saveAction = addMenuAction(menu, ":/icons/document-save.svg", "&Save Input File", "Ctrl+S",
                               &SpartaGui::save);
    addMenuAction(menu, ":/icons/document-save-as.svg", "Save Input File &As", "Ctrl+Shift+S",
                  &SpartaGui::saveAs);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/txt-file-icon.svg", "&View Text File", "Ctrl+Shift+F",
                  &SpartaGui::view);
    addMenuAction(menu, ":/icons/image-x-generic.svg", "View &Image or Movie File(s)...",
                  "Ctrl+Shift+J", &SpartaGui::openImages);
    // Ctrl+Shift+D ("data"): Ctrl+Shift+P went to the command palette, which
    // is what that binding means in every editor of this generation.
    addMenuAction(menu, ":/icons/x-office-drawing.svg", "&Plot Data File...", "Ctrl+Shift+D",
                  &SpartaGui::plotDataFile);
    addMenuAction(menu, ":/icons/binary-file-icon.svg", "Inspect &Restart File", "Ctrl+Shift+R",
                  &SpartaGui::inspect);
    restartAction = addMenuAction(menu, ":/icons/document-save.svg", "&Write Restart File...", "",
                                  &SpartaGui::writeRestart);
    menu->addSeparator();

    recentActions.resize(Cfg::NUM_RECENT_FILES);
    for (int i = 0; i < Cfg::NUM_RECENT_FILES; ++i) {
        recentActions[i] = addMenuAction(menu, ":/icons/document-open-recent.svg",
                                         QString("&%1.").arg(i + 1), "", &SpartaGui::openRecent);
        // their text is a file name, so the status-tip table finds them by this
        recentActions[i]->setObjectName(QString("recentfile%1").arg(i));
    }
    menu->addSeparator();

    addMenuAction(menu, ":/icons/application-exit.svg", "&Quit", "Ctrl+Q", &SpartaGui::quit);
}

void SpartaGui::createEditMenu()
{
    auto *menu = menubar->addMenu("&Edit");
    addMenuAction(menu, ":/icons/edit-undo.svg", "&Undo", "Ctrl+Z", &SpartaGui::undo);
    addMenuAction(menu, ":/icons/edit-redo.svg", "&Redo", "Ctrl+Shift+Z", &SpartaGui::redo);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/edit-copy.svg", "&Copy", "Ctrl+C", &SpartaGui::copy)
        ->setEnabled(hasClipboard);
    addMenuAction(menu, ":/icons/edit-cut.svg", "Cu&t", "Ctrl+X", &SpartaGui::cut)
        ->setEnabled(hasClipboard);
    addMenuAction(menu, ":/icons/edit-paste.svg", "&Paste", "Ctrl+V", &SpartaGui::paste)
        ->setEnabled(hasClipboard);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/vdw-style.svg", "Insert &Snippet...", "",
                  &SpartaGui::insertSnippet);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/search.svg", "&Find and Replace...", "Ctrl+F",
                  &SpartaGui::findAndReplace);
    menu->addSeparator();

    // The background input check finally gets a switch outside Preferences.
    // Checkable, so the menu itself shows whether it is on.
    auto *lintAction = addMenuAction(menu, ":/icons/warning.svg", "Autom&atic Input Checking", "",
                                     &SpartaGui::toggleAutoLint);
    lintAction->setCheckable(true);
    lintAction->setChecked(QSettings().value(Keys::AUTOLINT, true).toBool());
    autoLintAction = lintAction;
    menu->addSeparator();

    // On macOS Qt guesses an action's menu role from its text and moves anything
    // it reads as "preferences" into the application menu. Guessing is what goes
    // wrong here: both entries below have "Preferences" in them, so say which is
    // which rather than leaving it to a string match. Ctrl+P is also the wrong
    // key there -- it arrives as Cmd+P, which is Print -- so the entry that does
    // move gets the shortcut the platform expects for it.
    auto *prefsAction = addMenuAction(menu, ":/icons/preferences-desktop.svg", "P&references...",
                                      "Ctrl+P", &SpartaGui::preferences);
    prefsAction->setMenuRole(QAction::PreferencesRole);
#if defined(Q_OS_MACOS)
    prefsAction->setShortcut(QKeySequence::Preferences);
#endif
    auto *defaultsAction = addMenuAction(menu, ":/icons/preferences-reset.svg",
                                         "Reset Preferences to &Defaults", "",
                                         &SpartaGui::defaults);
    defaultsAction->setMenuRole(QAction::NoRole);
}

void SpartaGui::createRunMenu()
{
    auto *menu = menubar->addMenu("&Run");
    runAction = addMenuAction(menu, ":/icons/system-run.svg", "&Run SPARTA from Editor Buffer",
                              "Ctrl+Return", &SpartaGui::runBuffer);
    addMenuAction(menu, ":/icons/run-file.svg", "Run SPARTA from &File", "Ctrl+Shift+Return",
                  &SpartaGui::runFile);
    stopAction =
        addMenuAction(menu, ":/icons/process-stop.svg", "&Stop SPARTA", "Ctrl+/",
                      &SpartaGui::stopRun);
    extendAction = addMenuAction(menu, ":/icons/go-last.svg", "&Extend Run...", "Ctrl+E",
                                 &SpartaGui::extendRun);
    checkAction =
        addMenuAction(menu, ":/icons/warning.svg", "Chec&k Input", "Ctrl+K", &SpartaGui::checkInput);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/system-restart.svg", "Relaunch &SPARTA Instance", "",
                  &SpartaGui::restartSparta);
    menu->addSeparator();

    varsAction = addMenuAction(menu, ":/icons/preferences-desktop.svg", "Set &Variables...",
                               "Ctrl+Shift+V", &SpartaGui::editVariables);
    addMenuAction(menu, ":/icons/binary-file-icon.svg", "Insert &Restart Commands...", "",
                  &SpartaGui::continueRestart);
    // Create Image and the 3D snapshot used to end this menu; they moved to
    // Tools beside the other artifact-producing entries, leaving Run to the
    // run lifecycle alone.
}

void SpartaGui::createToolsMenu()
{
    auto *menu = menubar->addMenu("&Tools");
    // First because it finds everything else: type a few letters, see every
    // matching menu action with its shortcut, hit Enter.
    addMenuAction(menu, ":/icons/search.svg", "Command &Palette...", "Ctrl+Shift+P",
                  &SpartaGui::showPalette);
    menu->addSeparator();

    // Rendering: produces artifacts from the simulation state, same species
    // as the converters and reports below.
    imageAction = addMenuAction(menu, ":/icons/image-viewer.svg", "Create &Image", "Ctrl+I",
                                [this]() { renderImage(); });
#if defined(SPARTA_GUI_HAVE_VTK)
    addMenuAction(menu, ":/icons/image-viewer.svg", "3D &Snapshot (VTK)", "Ctrl+Shift+3",
                  &SpartaGui::renderVtkSnapshot);
#endif
    menu->addSeparator();

    // Geometry conversion, external export and reporting: work on simulation
    // data, but outside the edit-run-look loop that File and Run cover.
    addMenuAction(menu, ":/icons/vdw-style.svg", "Import Sur&face (STL / SPARTA)...", "Ctrl+Shift+T",
                  &SpartaGui::importSurface);
    addMenuAction(menu, ":/icons/image-x-generic.svg", "Export to Para&View...", "Ctrl+Shift+E",
                  &SpartaGui::exportParaview);
    addMenuAction(menu, ":/icons/vdw-style.svg", "Surface &Quantities Report...", "",
                  &SpartaGui::surfaceReport);
    menu->addSeparator();

    // Multi-run studies, directly here rather than in a "Studies" submenu: a
    // second level hid two features that already had no shortcuts, and with
    // the palette a submenu buys no tidiness worth that cost.
    addMenuAction(menu, ":/icons/x-office-drawing.svg", "Parametric S&weep...", "",
                  &SpartaGui::runSweep);
    addMenuAction(menu, ":/icons/document-open-recent.svg", "Run &History...", "",
                  &SpartaGui::showRunHistory);
}

void SpartaGui::createViewMenu()
{
    auto *menu = menubar->addMenu("&View");

    // Workspace modes come first: they are the primary way to change what the
    // window shows, and the individual panel toggles below are the escape hatch
    // for tailoring a mode.
    struct ModeEntry {
        PanelManager::Mode mode;
        const char *icon;
        const char *text;
        const char *shortcut;
    };
    static const ModeEntry modes[] = {
        {PanelManager::RunMode, ":/icons/system-run.svg", "&Run Workspace", "Ctrl+1"},
        {PanelManager::Analyze, ":/icons/x-office-drawing.svg", "&Analyze Workspace", "Ctrl+2"},
        {PanelManager::Visualize, ":/icons/image-viewer.svg", "&Visualize Workspace", "Ctrl+3"},
    };
    modeGroup = new QActionGroup(this);
    modeGroup->setExclusive(true);
    for (const auto &m : modes) {
        auto *act = addMenuAction(menu, m.icon, m.text, m.shortcut,
                                  [this, m]() { panels->applyMode(m.mode); });
        act->setCheckable(true);
        act->setData(int(m.mode));
        modeGroup->addAction(act);
    }
    menu->addSeparator();

    struct Entry {
        PanelManager::Panel panel;
        const char *icon;
        const char *text;
        const char *shortcut;
    };
    static const Entry entries[] = {
        {PanelManager::Log, ":/icons/utilities-terminal.svg", "&Output Window", "Ctrl+Shift+L"},
        {PanelManager::Chart, ":/icons/x-office-drawing.svg", "&Charts Window", "Ctrl+Shift+C"},
        {PanelManager::Viewer, ":/icons/image-viewer.svg", "&Viewer Window", "Ctrl+Shift+I"},
        {PanelManager::Variables, ":/icons/utilities-terminal.svg", "&Variables Window",
         "Ctrl+Shift+W"},
        {PanelManager::Sweep, ":/icons/x-office-drawing.svg", "Parametric S&weep Window", ""},
        {PanelManager::History, ":/icons/document-open-recent.svg", "Run &History Window", ""},
        {PanelManager::Diagnostics, ":/icons/warning.svg", "&Diagnostics Window",
         "Ctrl+Shift+X"},
        {PanelManager::ProjectFiles, ":/icons/document-open.svg", "Project &Files Window", ""},
    };
    for (const auto &e : entries) {
        auto *action = panels->toggleViewAction(e.panel);
        action->setIcon(QIcon(e.icon));
        panels->setPanelMenuText(e.panel, e.text);
        action->setShortcut(QKeySequence(e.shortcut));
        menu->addAction(action);
    }

    // persist only on user-driven toggles (QAction::triggered); run-driven
    // open/close of any panel (including the run-start viewer hide) must
    // not touch these settings
    connect(panels->toggleViewAction(PanelManager::Log), &QAction::triggered, this, [this]() {
        QSettings().setValue(Keys::VIEWLOG, panels->isPanelOpen(PanelManager::Log));
    });
    connect(panels->toggleViewAction(PanelManager::Chart), &QAction::triggered, this, [this]() {
        QSettings().setValue(Keys::VIEWCHART, panels->isPanelOpen(PanelManager::Chart));
    });

    // lazily (re-)create views whose widget is torn down by newDocument()/
    // openFile() if the user opens their panel again before the next run
    connect(panels, &PanelManager::panelOpened, this, [this](int panel) {
        if (panel == PanelManager::Viewer) ensureViewerPanel();
        // opening the viewer with no snapshot yet used to show an empty pane;
        // render one on demand (renderImage() creates the source and re-opens
        // the panel -- the no-snapshot-yet guard prevents recursion, and if
        // rendering is not possible it reports why instead of doing nothing).
        //
        // Not during startup: this same signal fires from restoreLayout(),
        // before the SPARTA plugin is even loaded, and rendering then would
        // call into an unloaded library and crash.
        //
        // And not while a run is going. The viewer panel is opened by the run
        // itself as soon as it writes its first frame, so auto-rendering here
        // would answer "Cannot create snapshot image while SPARTA is running"
        // with a modal dialog, in the middle of the run, without anyone having
        // asked for a snapshot.
        if (panel == PanelManager::Viewer && viewer && !viewer->snapshot() && startupComplete &&
            !sparta.isRunning())
            renderImage(/*quiet=*/true);
        if (panel == PanelManager::Log) ensureLogPanel();
        if (panel == PanelManager::Chart) ensureChartPanel();
        if (panel == PanelManager::Variables && !varwindow) createVariableWindow();
        if (panel == PanelManager::Sweep) ensureSweepPanel();
        if (panel == PanelManager::History) ensureHistoryPanel();
        if (panel == PanelManager::Diagnostics) ensureDiagnosticsPanel();
        if (panel == PanelManager::ProjectFiles) {
            ensureProjectFilesPanel();
            refreshProjectFiles();
        }
    });

    menu->addSeparator();
    // The viewer is one panel now, so these pick which of its pages is in
    // front. Ctrl+L still means "show me the frames" and Ctrl+Shift+I still
    // opens the viewer, which is what those keys did before the merge.
    addMenuAction(menu, ":/icons/image-viewer.svg", "S&napshot in Viewer", "", [this]() {
        ensureViewerPanel();
        panels->openPanel(PanelManager::Viewer);
        viewer->showSource(ViewerPanel::Snapshot, true);
    });
    addMenuAction(menu, ":/icons/image-x-generic.svg", "Slide S&how in Viewer", "Ctrl+L", [this]() {
        ensureViewerPanel();
        panels->openPanel(PanelManager::Viewer);
        // No need to build an empty slide show first: the panel's tab exists
        // whether or not a run has written a frame, and shows the card naming
        // the dump image command that would fill it.  That used to be worked
        // around here by constructing an empty SlideShow, which showed a blank
        // pane and explained nothing.
        viewer->showSource(ViewerPanel::Sequence, true);
    });
#if defined(SPARTA_GUI_HAVE_VTK)
    addMenuAction(menu, ":/icons/x-office-drawing.svg", "&3D Scene in Viewer", "", [this]() {
        ensureViewerPanel();
        panels->openPanel(PanelManager::Viewer);
        viewer->showSource(ViewerPanel::Scene, true);
    });
    addMenuAction(menu, ":/icons/image-viewer.svg", "3D Viewer &Window (VTK)", "",
                  &SpartaGui::open3DViewer);
#endif
    addMenuAction(menu, ":/icons/preferences-reset.svg", "Reset &Layout", "",
                  [this]() { panels->resetCurrentMode(); });
}

void SpartaGui::createAboutMenu()
{
    // "Help", not "About": that is where every desktop convention says these
    // live, and on macOS a menu with this title additionally gets the native
    // help-search field. The About entry keeps its AboutRole, so macOS still
    // relocates it into the application menu.
    auto *menu = menubar->addMenu("&Help");
    addMenuAction(menu, ":/icons/help-faq.svg", "Quick &Help", "Ctrl+Shift+H", &SpartaGui::help);
    // F1 through the platform mapping (Cmd+? on macOS), not hardcoded
    auto *keysAction = addMenuAction(menu, ":/icons/preferences-desktop-font.svg",
                                     "&Keyboard Shortcuts...", "", &SpartaGui::showShortcuts);
    keysAction->setShortcut(QKeySequence::HelpContents);
    menu->addSeparator();
    addMenuAction(menu, ":/icons/system-help.svg", "SPARTA-&GUI Documentation", "Ctrl+Shift+G",
                  &SpartaGui::howto);
    addMenuAction(menu, ":/icons/help-browser.svg", "SPARTA Online &Manual", "Ctrl+Shift+M",
                  &SpartaGui::manual);
    menu->addSeparator();
    auto *aboutAction = addMenuAction(menu, ":/icons/sparta-gui-icon-128x128.png",
                                      "&About SPARTA-GUI", "Ctrl+Shift+A", &SpartaGui::about);
    aboutAction->setMenuRole(QAction::AboutRole);

#if defined(SPARTA_GUI_USE_PLUGIN)
    menu->addSeparator();
    addMenuAction(menu, ":/icons/sparta-plugin.png", "Check for &SPARTA update", "Ctrl+Shift+U",
                  &SpartaGui::checkUpdate);
#endif
}

void SpartaGui::syncModeControls(int mode)
{
    if (modeGroup)
        for (QAction *a : modeGroup->actions()) a->setChecked(a->data().toInt() == mode);
    for (int i = 0; i < modeButtons.size(); ++i) modeButtons[i]->setChecked(i == mode);
}


void SpartaGui::createToolBar()
{
    // The primary actions get a first-class, labeled, always-visible surface.
    // They used to live as four icon-only buttons inside the *status bar* --
    // which is for output, not input -- and everything else was menu-only.
    // Every button here is the same QAction as its menu entry, so shortcuts,
    // enabled state and checkmarks agree everywhere by construction.
    auto *tb = new QToolBar("Main Toolbar", this);
    tb->setObjectName("maintoolbar");
    tb->setMovable(false);
    tb->setFloatable(false);
    tb->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    // no "hide the toolbar" context-menu foot-gun: with the status-bar buttons
    // gone this is where the primary actions live
    tb->toggleViewAction()->setVisible(false);
    addToolBar(Qt::TopToolBarArea, tb);

    // short labels for the toolbar; the menus keep the full text
    newAction->setIconText("New");
    openAction->setIconText("Open");
    saveAction->setIconText("Save");
    runAction->setIconText("Run");
    stopAction->setIconText("Stop");
    checkAction->setIconText("Check");
    varsAction->setIconText("Variables");
    imageAction->setIconText("Image");

    tb->addAction(newAction);
    tb->addAction(openAction);
    // Open doubles as the gateway to the bundled examples: its dropdown is
    // the same Open Example submenu the File menu holds.
    if (auto *openBtn = qobject_cast<QToolButton *>(tb->widgetForAction(openAction))) {
        openBtn->setMenu(exampleMenu);
        openBtn->setPopupMode(QToolButton::MenuButtonPopup);
    }
    tb->addAction(saveAction);
    tb->addSeparator();
    tb->addAction(runAction);
    tb->addAction(stopAction);
    tb->addAction(checkAction);
    tb->addSeparator();
    tb->addAction(varsAction);
    tb->addAction(imageAction);

    // Everything after this spacer sits on the right edge.
    auto *spacer = new QWidget(tb);
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    tb->addWidget(spacer);

    // Workspace mode switch: a segmented control of checkable buttons. It
    // decides what the window shows, so it belongs in the toolbar -- with the
    // controls -- not in the status bar it used to sit in.
    struct ModeBtn {
        PanelManager::Mode mode;
        const char *text;
        const char *tip;
    };
    static const ModeBtn modebtns[] = {
        {PanelManager::RunMode, "Run", "Prepare a deck and watch it run: console output and variables"},
        {PanelManager::Analyze, "Analyze", "Study results: the charts, full size"},
        {PanelManager::Visualize, "Visualize", "Look at the pictures with the window given over to them"},
    };
    auto *modebar = new QWidget(tb);
    auto *modelay = new QHBoxLayout(modebar);
    modelay->setContentsMargins(0, 0, 0, 0);
    modelay->setSpacing(0);
    for (const auto &m : modebtns) {
        auto *b = new QPushButton(m.text, modebar);
        b->setCheckable(true);
        b->setToolTip(m.tip);
        b->setProperty("spartaModeButton", true);
        connect(b, &QPushButton::clicked, this, [this, m]() { panels->applyMode(m.mode); });
        modelay->addWidget(b);
        modeButtons.append(b);
    }
    tb->addWidget(modebar);

    auto *paletteBtn = new QToolButton(tb);
    paletteBtn->setIcon(QIcon(":/icons/search.svg"));
    paletteBtn->setToolTip("Command palette: search every menu action (Ctrl+Shift+P)");
    paletteBtn->setAccessibleName("Command palette");
    connect(paletteBtn, &QToolButton::clicked, this, &SpartaGui::showPalette);
    tb->addWidget(paletteBtn);

    syncRunControls();
}

void SpartaGui::createStatusBar()
{
    statusbar = new QStatusBar(this);
    setStatusBar(statusbar);

    spartastatus = new QLabel(QString());
    auto pix     = QPixmap(Cfg::SPARTA_ICON);
    spartastatus->setPixmap(pix.scaled(Cfg::ICON_SCALE, Cfg::ICON_SCALE, Qt::KeepAspectRatio));
    spartastatus->setToolTip("SPARTA instance is active");
    spartastatus->hide();
    statusbar->addWidget(spartastatus);

    cpuuse = new QLabel(Cfg::STATUS_ZERO_CPU);
    cpuuse->setFixedWidth(90);
    statusbar->addWidget(cpuuse);
    cpuuse->hide();

    // The status bar sets the floor for how narrow the window can get, so its
    // labels are elastic rather than fixed: between them a 300px status, a
    // 400px directory label and a 400px progress bar demanded over 1100px of
    // width before any panel had a say. Give each a modest minimum and let the
    // layout distribute what is actually available, eliding the directory path
    // (from the left, so the interesting trailing component survives).
    status = new QLabel(Cfg::STATUS_READY);
    status->setMinimumWidth(Cfg::STATUS_LABEL_MIN_WIDTH);
    status->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    statusbar->addWidget(status);

    dirstatus = new ElidedLabel(QString(" Directory: (unknown)"));
    dirstatus->setMinimumWidth(Cfg::STATUS_LABEL_MIN_WIDTH);
    dirstatus->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    dirstatus->show();
    statusbar->addWidget(dirstatus, 1);

    progress = new QProgressBar();
    progress->setRange(0, Cfg::PROGRESS_MAXIMUM);
    progress->setMinimumWidth(Cfg::PROGRESS_MIN_WIDTH);
    progress->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    progress->hide();
    statusbar->addWidget(progress);
}

#if defined(SPARTA_GUI_USE_PLUGIN)
void SpartaGui::setupPlugin(QSettings &settings)
{
    // A way to reach the no-library state on a machine that has one, so the
    // setup card is testable without hiding the user's library from them.
    if (qgetenv("SPARTA_GUI_FORCE_NO_PLUGIN") == "1") {
        pluginPath.clear();
        return;
    }

    // first try to load from existing setting
    pluginPath = settings.value(Keys::PLUGIN_PATH, "").toString();
    if (!pluginPath.isEmpty()) {
        // make canonical and try loading; reset to empty string if loading failed
        pluginPath = QFileInfo(pluginPath).canonicalFilePath();
        if (!sparta.loadLib(pluginPath)) {
            pluginPath.clear();
            // could not load successfully -> remove any existing setting.
            settings.remove(Keys::PLUGIN_PATH);
        }
    }
    if (!pluginPath.isEmpty()) return;

    // Nothing configured, or what was configured no longer loads: try every
    // library-shaped file this platform puts within reach, most specific first.
    for (const auto &libpath : LibraryAcquire::candidates()) {
        if (sparta.loadLib(libpath)) {
            pluginPath = libpath;
            settings.setValue(Keys::PLUGIN_PATH, pluginPath);
            settings.sync();
            return;
        }
    }

    // Still nothing.  This is where a modal dialog used to take over and
    // refuse to let the application start; the setup card takes it from here,
    // and the caller carries on without a simulator.
    settings.remove(Keys::PLUGIN_PATH);
}
#else
// dummy function when linking against library directly
void SpartaGui::setupPlugin(QSettings &) {}
#endif

void SpartaGui::browseForLibrary()
{
    const QString chosen = QFileDialog::getOpenFileName(
        this, "Select SPARTA shared library to use", ".", LibraryAcquire::fileDialogPattern(),
        nullptr, QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly);
    if (chosen.isEmpty()) return; // cancelling is a choice, not a failure

    // The name check is a heuristic and the user is allowed to overrule it;
    // whether the file loads is the real test.
    if (!LibraryAcquire::nameLooksRight(chosen)) {
        if (QMessageBox::question(
                this, "SPARTA-GUI - Unexpected File Name",
                QString("The name of\n\n%1\n\ndoes not contain \"libsparta\". "
                        "SPARTA-GUI expects the SPARTA shared library, e.g. "
                        "libsparta.so or libsparta.dylib.\n\n"
                        "Try to load this file anyway?")
                    .arg(QDir::toNativeSeparators(chosen)),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes)
            return;
    }

    if (!adoptLibrary(chosen))
        setupCard->setError(
            QString("%1 is not a SPARTA shared library this build can load (wrong platform, "
                    "architecture, or build).")
                .arg(QDir::toNativeSeparators(chosen)));
}

void SpartaGui::downloadLibrary()
{
    const QString dest = LibraryAcquire::downloadDestination();
    if (dest.isEmpty()) {
        setupCard->setError("Cannot create a writable directory in the user configuration folder "
                            "to store the downloaded library in.");
        return;
    }

    QString reason;
    switch (LibraryAcquire::download(this, dest, &reason)) {
        case LibraryAcquire::Result::Cancelled:
            return;
        case LibraryAcquire::Result::Failed:
            setupCard->setError("Downloading the SPARTA shared library failed: " + reason);
            return;
        case LibraryAcquire::Result::Ok:
            break;
    }

    if (!adoptLibrary(dest)) {
        QFile::remove(dest);
        setupCard->setError("The downloaded shared library does not seem to be compatible with "
                            "this system.");
    }
}

bool SpartaGui::adoptLibrary(const QString &path)
{
    const QString canonical = QFileInfo(path).canonicalFilePath();
    if (canonical.isEmpty() || !sparta.loadLib(canonical)) return false;

    pluginPath = canonical;
    QSettings settings;
    settings.setValue(Keys::PLUGIN_PATH, pluginPath);
    settings.sync();

    // Nothing was loaded before this, so the library can simply be used from
    // here -- no relaunch.  (Replacing an already-loaded library, which is what
    // Preferences does, is the case that needs one.)
    finishLibraryInit(settings);
    syncSetupCard();
    return true;
}

void SpartaGui::syncSetupCard()
{
    if (!setupCard) return;
    const bool missing = !sparta.hasLibrary();
    setupCard->setVisible(missing);
    if (!missing) setupCard->clearError();
    syncRunControls();
}

void SpartaGui::finishLibraryInit(QSettings &settings)
{
    setupAccelerators(settings);

    // again: the library's directory is one of the places examples are looked
    // for, and it was not known when the constructor first built this
    buildExampleMenu();

    // start SPARTA and initialize command completion
    startSparta();

    // the command list is the built-in commands plus the command styles this
    // library was built with; without a library it is the built-ins alone,
    // which is why the internal list is loaded whether or not we get here
    QStringList style_list;
    QFile internal_commands(":/sparta_internal_commands.txt");
    if (internal_commands.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!internal_commands.atEnd())
            style_list << QString(internal_commands.readLine()).trimmed();
    }
    internal_commands.close();

    const int ncmds = sparta.styleCount("command");
    for (int i = 0; i < ncmds; ++i) {
        const QString style = sparta.styleName("command", i);
        if (style.isEmpty()) continue;
        // skip suffixed names
        if (style.endsWith("/kk/host") || style.endsWith("/kk/device") || style.endsWith("/kk"))
            continue;
        style_list << style;
    }
    style_list.sort();
    textEdit->setCommandList(style_list);

    // build a sorted, accelerator-suffix-filtered style list for one category
    auto styleList = [&](const char *keyword) {
        QStringList list;
        const int nstyles = sparta.styleCount(keyword);
        for (int i = 0; i < nstyles; ++i) {
            const QString style = sparta.styleName(keyword, i);
            if (style.isEmpty()) continue;
            if (style.endsWith("/kk") || style.endsWith("/kk/device") ||
                style.endsWith("/kk/host"))
                continue;
            list << style;
        }
        list.sort();
        return list;
    };

    textEdit->setFixList(styleList("fix"));
    textEdit->setComputeList(styleList("compute"));
    textEdit->setDumpList(styleList("dump"));
    textEdit->setRegionList(styleList("region"));
    textEdit->setCollideList(styleList("collide"));
    textEdit->setReactList(styleList("react"));
    textEdit->setSurfCollideList(styleList("surf_collide"));
    textEdit->setSurfReactList(styleList("surf_react"));

    // apply https proxy setting: prefer environment variable or fall back to
    // preferences value
    applyProxySetting(sparta, settings);
}

void SpartaGui::setupAccelerators(QSettings &settings)
{
    // SPARTA only supports the KOKKOS accelerator package. Switch the configured
    // accelerator to "none" if KOKKOS is not available so there is always a
    // working option.
    int accel = settings.value(Keys::ACCELERATOR, AcceleratorTab::None).toInt();
    if ((accel != AcceleratorTab::Kokkos) || !sparta.configHasPackage("KOKKOS"))
        accel = AcceleratorTab::None;
    settings.setValue(Keys::ACCELERATOR, accel);

    // Check and initialize nthreads setting for when OpenMP support is compiled in.
    // Default is to use OMP_NUM_THREADS setting, if that is not available, then half of max
    // (assuming hyper-threading is enabled) and no more than Cfg::MAX_DEFAULT_THREADS
    // (=16). This is only if there is no preference set but do not override OMP_NUM_THREADS
    int default_threads = std::min(QThread::idealThreadCount() / 2, Cfg::MAX_DEFAULT_THREADS);
    default_threads     = std::max(default_threads, 1);
    if (qEnvironmentVariableIsSet("OMP_NUM_THREADS"))
        default_threads = qEnvironmentVariable("OMP_NUM_THREADS").toInt();
    nthreads = settings.value(Keys::NTHREADS, default_threads).toInt();

    // reset nthreads if accelerator does not support threads
    if (accel == AcceleratorTab::None) nthreads = 1;

    // set OMP_NUM_THREADS environment variable, if not set
    if (!qEnvironmentVariableIsSet("OMP_NUM_THREADS"))
        qputenv("OMP_NUM_THREADS", QByteArray::number(nthreads));
}

/* -------------------------------------------------------------------- */

SpartaGui::SpartaGui(QWidget *parent, const QString &filename, int width, int height) :
    QMainWindow(parent), textEdit(nullptr), centralStack(nullptr), welcome(nullptr),
    menubar(nullptr), exampleMenu(nullptr),
    highlighter(nullptr), capturer(new StdCapture), status(nullptr), cpuuse(nullptr),
    lastCpuBucket(-1), panels(nullptr), logwindow(nullptr), viewer(nullptr),
    chartwindow(nullptr), logupdater(nullptr), dirstatus(nullptr),
    progress(nullptr),
    prefdialog(nullptr), spartastatus(nullptr), varwindow(nullptr), runner(nullptr),
    runCounter(0), extendSteps(Cfg::EXTEND_STEPS_DEFAULT), nthreads(1), mainx(width),
    mainy(height)
{
#if QT_CONFIG(clipboard)
    hasClipboard = true;
#else
    hasClipboard = false;
#endif

#if !defined(Q_OS_MACOS)
    // minimize window so we don't see it while it is being constructed and configured.
    // this hack does not work as expected on macOS but it is also not really needed.
    showMinimized();
#endif

    // restore and initialize settings
    QSettings settings;

    // configure fonts
    QFont allFont;
    QFontInfo allInfo(*GUI_ALLFONT);
    allFont.setFamily(settings.value(Keys::ALLFAMILY, allInfo.family()).toString());
    allFont.setPointSize(settings.value(Keys::ALLSIZE, allInfo.pointSize()).toInt());
    allFont.setStyleHint(GUI_ALLFONT->styleHint());
    settings.setValue(Keys::ALLFAMILY, allFont.family());
    settings.setValue(Keys::ALLSIZE, allFont.pointSize());

    QFont monoFont = monoFontFromSettings();
    settings.setValue(Keys::MONOFAMILY, monoFont.family());
    settings.setValue(Keys::MONOSIZE, monoFont.pointSize());

    // create and connect GUI elements
    setupUi(settings, allFont, monoFont);
    // Before the layout is applied, not after: a workspace only opens panels
    // that already hold a widget, so the Output dock has to have one by now or
    // the Setup workspace comes up as a bare editor.
    ensureLogPanel();
    // fall back to the built-in default layout if there is no saved state yet
    // (first launch) or it doesn't match the current DOCK_LAYOUT_VERSION
    restoredLayout = panels->restoreLayout(settings);

    currentFile.clear();
    currentDir = QDir(".").absolutePath();
    // use $HOME if we get dropped to "/" like on macOS or the installation folder or
    // system folder like on Windows
    if ((currentDir == "/") || (currentDir.contains("AppData")) ||
        (currentDir.contains("system32")))
        currentDir = QDir::homePath();
    QDir::setCurrent(currentDir);
    dirstatus->setText(QString(" Directory: ") + currentDir);

    setAutoFillBackground(true);

    setupPlugin(settings);

    // Examples are files, so they can be opened and read without a simulator.
    // finishLibraryInit() builds this again once there is one, because the
    // library's own location is one more place examples are looked for.
    buildExampleMenu();

    // set up default SPARTA thread arguments
    spartaArgs.clear();
    spartaArgs.push_back("SPARTA-GUI");
    spartaArgs.push_back("-log");
    spartaArgs.push_back("none");

    installEventFilter(this);

    settings.sync();

    updateRecents();

    if ((filename.size() > 0) && !filename.endsWith("sparta-gui.exe")) {
        openFile(filename);
    } else {
        setWindowTitle("SPARTA-GUI - Editor - *unknown*");
        // restore the previous session's window geometry
        const QByteArray geo = settings.value(Keys::WINGEOMETRY).toByteArray();
        if (!geo.isEmpty()) restoreGeometry(geo);

        // offer to recover a buffer left by a previous crash; otherwise reopen
        // the last session's file; otherwise greet with the welcome screen
        bool opened = maybeRecoverSession();
        if (!opened && settings.value(Keys::RESTORE_SESSION, true).toBool()) {
            const QString last = settings.value(Keys::LAST_FILE).toString();
            if (!last.isEmpty() && QFileInfo::exists(last)) {
                openFile(last);
                opened = true;
            }
        }
        if (!opened && settings.value(Keys::SHOWWELCOME, true).toBool()) showWelcome();
    }
    startRecoveryTimer();

    // The command list, so completion works on a deck the user is reading even
    // when there is no library to add this build's command styles to it.
    QStringList style_list;
    QFile internal_commands(":/sparta_internal_commands.txt");
    if (internal_commands.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!internal_commands.atEnd()) {
            style_list << QString(internal_commands.readLine()).trimmed();
        }
    }
    internal_commands.close();
    style_list.sort();
    textEdit->setCommandList(style_list);

    style_list.clear();
    const char *varstyles[] = {"delete", "equal",  "file",     "format", "getenv", "grid",
                               "index",  "internal", "loop",   "particle", "python", "string",
                               "surf",   "uloop",  "universe", "world"};
    for (const auto *const var : varstyles)
        style_list << var;
    style_list.sort();
    textEdit->setVariableList(style_list);

    style_list.clear();
    const char *unitstyles[] = {"si", "cgs"};
    for (const auto *const unit : unitstyles)
        style_list << unit;
    style_list.sort();
    textEdit->setUnitsList(style_list);

    textEdit->setFileList();

    settings.beginGroup(Keys::GROUP_REFORMAT);
    textEdit->setReformatOnReturn(settings.value(Keys::RETURN, false).toBool());
    textEdit->setAutoComplete(settings.value(Keys::AUTOMATIC, true).toBool());
    autoLintEnabled = settings.value(Keys::AUTOLINT, true).toBool();
    settings.endGroup();

    // Everything from here needs a loaded library.  Without one the editor is
    // fully usable and the setup card says what is missing and offers the two
    // ways out; finishLibraryInit() runs the moment one of them works.
    if (sparta.hasLibrary())
        finishLibraryInit(settings);
    else
        applyProxySetting(sparta, settings); // the download needs the proxy too
    syncSetupCard();

    // finally show the window
    showNormal();

    // the UI is now fully built and the SPARTA plugin loaded: from here on a
    // panelOpened(Image) is a real user action and may auto-render a snapshot
    // (see the PanelManager::panelOpened handler in createViewMenu()).
    startupComplete = true;

    // Populate the workspace the session opens in. The mode was applied while
    // restoring the layout, before the plugin was loaded, so its panels could
    // not be created then; do it now that everything is in place. Only for a
    // fresh session -- a restored layout already describes what was open.
    if (!restoredLayout) panels->applyMode(panels->currentMode());
}

SpartaGui::~SpartaGui()
{
    // The restart-inspection windows first: they hold a pointer to `sparta`,
    // which is a member of this object.  Left to Qt they would be destroyed by
    // ~QWidget as children -- after the members are gone -- and the Hide event
    // that reaches their event filter on the way out would call the simulator
    // through a wrapper that no longer exists.  purgeInspectList() only
    // collects the ones already hidden, which is right while the window is
    // alive and wrong now, so this takes them regardless.
    for (auto *item : inspectList) {
        delete item->info;
        delete item->image;
        delete item;
    }
    inspectList.clear();

    delete highlighter;
    delete capturer;
    delete status;
    delete cpuuse;
    delete dirstatus;
}

void SpartaGui::newDocument()
{
    // prompt to save unsaved changes before discarding the buffer, matching
    // the behavior of Open and Run (which already guard this)
    if (textEdit->document()->isModified()) {
        int rv = showUnsavedChangesDialog(
            this, currentFile, "Do you want to save the file before starting a new one?");
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel:
                return;
            case QMessageBox::No: // fallthrough
            default:
                break;
        }
    }

    currentFile.clear();
    textEdit->document()->setPlainText(citeme);
    textEdit->document()->setModified(false);
    applyEditorColorScheme();

    stopAndReapRunner();
    // close windows.  clearRunPanels() deletes *every* docked panel widget, so
    // every raw pointer we keep to one must be cleared here or it dangles (and
    // e.g. the auto-lint timer would then clear() a freed diagnostics list --
    // a crash).  Stop the pending auto-lint too so it cannot fire mid-teardown.
    if (autoLintTimer) autoLintTimer->stop();
    clearPanelWidgets();

    {
        StdoutSilencer guard;
        sparta.close();
    }
    spartastatus->hide();
    setWindowTitle("SPARTA-GUI - Editor - *unknown*");
    runCounter = 0;
    showEditor();
}

void SpartaGui::open()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open the file", QString(), "SPARTA input files (in.*);;All files (*)");
    openFile(fileName);
}

void SpartaGui::view()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open the file");
    viewFile(fileName);
}

void SpartaGui::inspect()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open the restart file");
    inspectFile(fileName);
}

void SpartaGui::openRecent()
{
    auto *act = qobject_cast<QAction *>(sender());
    if (act) openFile(act->data().toString());
}

// locate the folder with the SPARTA example inputs. The preferences setting has
// priority; otherwise probe folders relative to the plugin library and upward
// from the current working directory for an "examples" folder that contains
// subfolders with in.* input files.
QString SpartaGui::findExamplesDir() const
{
    auto isExamplesDir = [](const QString &path) {
        if (path.isEmpty()) return false;
        QDir dir(path);
        if (!dir.exists()) return false;
        const auto subdirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const auto &sub : subdirs) {
            if (!QDir(sub.absoluteFilePath())
                     .entryList({QStringLiteral("in.*")}, QDir::Files)
                     .isEmpty())
                return true;
        }
        return false;
    };

    QSettings settings;
    const QString configured = settings.value(Keys::EXAMPLES_PATH, "").toString();
    if (isExamplesDir(configured)) return QFileInfo(configured).canonicalFilePath();

    QStringList candidates;
    // examples bundled inside the application (macOS .app/Contents/Resources,
    // or a Linux/Windows install tree) are the primary location
    const QString appdir = QCoreApplication::applicationDirPath();
    candidates << appdir + "/../Resources/examples"     // macOS app bundle
               << appdir + "/../share/sparta/examples"  // Linux/Windows install
               << appdir + "/examples";
    if (!pluginPath.isEmpty()) {
        QDir libdir = QFileInfo(pluginPath).absoluteDir();
        candidates << libdir.absoluteFilePath("../../examples")
                    << libdir.absoluteFilePath("../../../examples");
    }
    QDir dir(QDir::currentPath());
    do {
        candidates << dir.absoluteFilePath("examples");
    } while (dir.cdUp());

    for (const auto &candidate : std::as_const(candidates))
        if (isExamplesDir(candidate)) return QFileInfo(candidate).canonicalFilePath();
    return {};
}

void SpartaGui::buildExampleMenu()
{
    if (!exampleMenu) return;
    exampleMenu->clear();

    const QString dirname = findExamplesDir();
    if (!dirname.isEmpty()) {
        QDir exdir(dirname);
        const auto subdirs = exdir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        for (const auto &sub : subdirs) {
            // the benchmark inputs are not instructive examples
            if (sub.fileName() == "bench") continue;
            const auto inputs = QDir(sub.absoluteFilePath())
                                    .entryInfoList({QStringLiteral("in.*")}, QDir::Files, QDir::Name);
            if (inputs.isEmpty()) continue;
            auto *submenu = exampleMenu->addMenu(sub.fileName());
            for (const auto &input : inputs) {
                auto *action = submenu->addAction(input.fileName());
                action->setData(input.absoluteFilePath());
                connect(action, &QAction::triggered, this, &SpartaGui::openExample);
            }
        }
    }
    exampleMenu->setEnabled(!exampleMenu->isEmpty());

    // (Re-)apply the status tips over the finished menu tree. Here rather
    // than in the constructor because this menu is the one part of the tree
    // that is rebuilt at runtime (after a preferences change), and the fresh
    // entries need their tips again.
    applyActionMetadata(menubar);
}

void SpartaGui::openExample()
{
    auto *act = qobject_cast<QAction *>(sender());
    if (!act) return;
    openExamplePath(act->data().toString());
}

void SpartaGui::openExamplePath(const QString &srcfile)
{
    if (srcfile.isEmpty()) return;
    QFileInfo srcinfo(srcfile);
    const QString srcdir = srcinfo.absolutePath();

    // examples are usually bundled read-only (e.g. inside the macOS .app), so
    // copy the whole example directory (input script plus its data files) into
    // a writable location the first time and open the copy from there, so the
    // simulation can actually be run and write its log and image output
    if (QFileInfo(srcdir).isWritable()) {
        openFile(srcfile);
        return;
    }

    const QString subname  = QFileInfo(srcdir).fileName();
    QDir destroot(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) +
                  "/SPARTA-GUI Examples");
    const QString destdir  = destroot.absoluteFilePath(subname);
    const QString destfile = destdir + "/" + srcinfo.fileName();

    if (!QFileInfo::exists(destfile)) {
        if (!destroot.mkpath(subname)) {
            warning(this, "Open Example", "Could not create a writable copy of the example in:",
                    destdir);
            return;
        }
        // copy the input scripts and data files, but not any subdirectories
        const auto entries = QDir(srcdir).entryInfoList(QDir::Files);
        for (const auto &entry : entries)
            QFile::copy(entry.absoluteFilePath(), destdir + "/" + entry.fileName());
    }
    openFile(destfile);
    // Say so: the file now open is not the one the user clicked, and edits land
    // in the copy. Everything a run writes goes there too. Non-modal on purpose
    // -- this is information, not a decision.
    statusBar()->showMessage(
        QString("Example copied to the writable folder %1 - editing and running the copy.")
            .arg(QDir::toNativeSeparators(destdir)),
        8000);
}

void SpartaGui::showWelcome()
{
    welcome->setRecentFiles(recent);
    welcome->setExamplesDir(findExamplesDir());
    // the run panels (Output, Charts, Image, ...) only hold simulation output
    // and would show up as empty docks stealing space on the welcome screen, so
    // hide them all and let the welcome page use the full window. A subsequent
    // run re-opens whichever panels the View settings call for.
    for (int i = 0; i < PanelManager::NPanels; ++i)
        panels->closePanel(static_cast<PanelManager::Panel>(i));
    centralStack->setCurrentWidget(welcome);
    // not "Editor - *unknown*": no editor is on screen and nothing is unknown
    setWindowTitle("SPARTA-GUI - Welcome");
}

void SpartaGui::showEditor()
{
    centralStack->setCurrentWidget(textEdit);
    // restore the editor title the welcome page replaced
    setWindowTitle(currentFile.isEmpty() ? QString("SPARTA-GUI - Editor - *unknown*")
                                         : QString("SPARTA-GUI - Editor - " + currentFile));
    if (textEdit->document()->isModified()) modified();
}

void SpartaGui::updateRecents(const QString &filename)
{
    QSettings settings;
    if (settings.contains(Keys::RECENT))
        recent = settings.value(Keys::RECENT).value<QList<QString>>();

    recent.removeIf([](const QString &f) {
        return !QFileInfo(f).isReadable();
    });

    if (!filename.isEmpty() && !recent.contains(filename)) recent.prepend(filename);
    if (recent.size() > Cfg::NUM_RECENT_FILES) recent.removeLast();
    if (!recent.empty())
        settings.setValue(Keys::RECENT, QVariant::fromValue(recent));
    else
        settings.remove(Keys::RECENT);

    for (int i = 0; i < Cfg::NUM_RECENT_FILES; ++i) {
        recentActions[i]->setVisible(false);
        if (i < recent.size() && !recent[i].isEmpty()) {
            QFileInfo fi(recent[i]);
            recentActions[i]->setText(QString("&%1. ").arg(i + 1) + fi.fileName());
            recentActions[i]->setData(recent[i]);
            recentActions[i]->setVisible(true);
        }
    }
}

// delete all current variables in the SPARTA instance
void SpartaGui::clearVariables()
{
    int nvar = sparta.idCount("variable");

    // delete from back so they are not re-indexed
    for (int i = nvar - 1; i >= 0; --i) {
        const QString name = sparta.idName("variable", i);
        if (!name.isEmpty()) sparta.command(QString("variable %1 delete").arg(name));
    }
}

void SpartaGui::updateVariables(bool keepOverrides)
{
    // what the user has set so far, so a re-scan of the same buffer does not
    // throw it away
    QHash<QString, QString> previous;
    QList<QPair<QString, QString>> previousList;
    if (keepOverrides) {
        previousList = variables;
        for (const auto &v : std::as_const(variables))
            previous.insert(v.first, v.second);
    }

    const auto doc = textEdit->toPlainText().replace('\t', ' ').split('\n');
    QStringList known;
    QRegularExpression indexvar(R"(^\s*variable\s+(\w+)\s+index\s+(.*))");
    QRegularExpression anyvar(R"(^\s*variable\s+(\w+)\s+(\w+)\s+(.*))");
    QRegularExpression usevar(R"((\$(\w)|\${(\w+)}))");
    QRegularExpression refvar(R"(v_(\w+))");

    // forget previously listed variables
    variables.clear();
    scriptVariables.clear();

    for (const auto &line : doc) {

        if (line.isEmpty()) continue;

        // first find variable definitions.
        // index variables are special since they can be overridden from the command line
        auto index = indexvar.match(line);
        auto any   = anyvar.match(line);

        if (index.hasMatch()) {
            if (index.lastCapturedIndex() >= 2) {
                auto name              = index.captured(1);
                const QString deckValue = index.captured(2);
                if (!known.contains(name)) {
                    // the deck's own value is recorded either way; the value
                    // offered for editing is the user's if they set one
                    scriptVariables.insert(name, deckValue);
                    variables.append(
                        qMakePair(name, previous.value(name, deckValue)));
                    known.append(name);
                }
            }
        } else if (any.hasMatch()) {
            if (any.lastCapturedIndex() >= 3) {
                auto name = any.captured(1);
                if (!known.contains(name)) known.append(name);
            }
        }

        // now split line into words and search for use of undefined variables
        auto words = line.split(' ', Qt::SkipEmptyParts);
        for (const auto &word : words) {
            auto use = usevar.match(word);
            auto ref = refvar.match(word);
            if (use.hasMatch()) {
                auto name = use.captured(use.lastCapturedIndex());
                if (!known.contains(name)) {
                    known.append(name);
                    variables.append(qMakePair(name, QString()));
                }
            }
            if (ref.hasMatch()) {
                auto name = ref.captured(ref.lastCapturedIndex());
                if (!known.contains(name)) known.append(name);
            }
        }
    }

    // Variables the user added by hand are not in the deck, so the scan above
    // cannot find them -- but they are still theirs, and are passed to the run
    // the way -var does on the command line.  Dropping them here would quietly
    // undo an edit made in the Set Variables dialog.
    if (keepOverrides)
        for (const auto &v : std::as_const(previousList))
            if (!known.contains(v.first)) variables.append(v);
}

// open file and switch CWD to path of file
void SpartaGui::openFile(const QString &fileName)
{
    // do nothing, if no file name provided
    if (fileName.isEmpty()) return;

    stopAndReapRunner();
    // close windows.  clearRunPanels() deletes *every* docked panel widget, so
    // every raw pointer we keep to one must be cleared here or it dangles (and
    // e.g. the auto-lint timer would then clear() a freed diagnostics list --
    // a crash).  Stop the pending auto-lint too so it cannot fire mid-teardown.
    if (autoLintTimer) autoLintTimer->stop();
    clearPanelWidgets();
    {
        StdoutSilencer guard;
        sparta.close();
    }

    purgeInspectList();
    textEdit->setStyleSheet("");
    if (textEdit->document()->isModified()) {
        int rv = showUnsavedChangesDialog(
            this, currentFile, "Do you want to save the file before opening a new file?");
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel:
                return;
            case QMessageBox::No: // fallthrough
            default:
                // do nothing
                break;
        }
    }
    textEdit->setHighlight(CodeEditor::NO_HIGHLIGHT, false);

    QFileInfo path(fileName);
    currentFile = path.fileName();
    currentDir  = path.absolutePath();
    QFile file(path.absoluteFilePath());

    updateRecents(path.absoluteFilePath());

    QDir::setCurrent(currentDir);
    if (!file.open(QIODevice::ReadOnly | QFile::Text)) {
        warning(this, "SPARTA-GUI Warning", "Cannot open file " + path.absoluteFilePath() + ":",
                file.errorString() + "\n\nWill create new file on saving editor buffer.");
        textEdit->document()->clear();
        textEdit->document()->setPlainText(citeme);
        textEdit->document()->setModified(false);
        applyEditorColorScheme();
    } else {
        QTextStream in(&file);
        QString text = in.readAll();
        textEdit->document()->clear();
        textEdit->document()->setPlainText(text);
        textEdit->moveCursor(QTextCursor::Start, QTextCursor::MoveAnchor);
        file.close();
    }
    setWindowTitle(QString("SPARTA-GUI - Editor - " + currentFile));
    runCounter = 0;
    textEdit->document()->setModified(false);
    textEdit->setGroupList();
    textEdit->setVarNameList();
    textEdit->setComputeIDList();
    textEdit->setFixIDList();
    textEdit->setMixtureIDList();
    textEdit->setFileList();
    dirstatus->setText(QString(" Directory: ") + currentDir);
    status->setText(Cfg::STATUS_READY);
    cpuuse->hide();

    updateVariables();
    if (projectFilesList) refreshProjectFiles();
    showEditor();
}

// open file in read-only mode for viewing in separate window
void SpartaGui::viewFile(const QString &fileName)
{
    // empty name means the file dialog was cancelled: nothing to view
    if (fileName.isEmpty()) return;

    // a movie file is also an image file when it is an animated GIF
    if (isMovieFile(fileName)) {
        warning(this, "Cannot View Movie as Text",
                "\"" + QFileInfo(fileName).fileName() +
                    "\" is a movie file and cannot be displayed in the text viewer.\n"
                    "Use \"View Image or Movie File(s)...\" (Ctrl+Shift+J) to open it.");
        return;
    }

    if (isImageFile(fileName)) {
        warning(this, "Cannot View Image as Text",
                "\"" + QFileInfo(fileName).fileName() +
                    "\" is an image file and cannot be displayed in the text viewer.\n"
                    "Use \"View Image or Movie File(s)...\" (Ctrl+Shift+J) to open it.");
        return;
    }

    if (looksLikeBinaryFile(fileName)) {
        warning(this, "Cannot View Binary File as Text",
                "\"" + QFileInfo(fileName).fileName() +
                    "\" appears to be a binary file and cannot be displayed in the text viewer.");
        return;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QFile::Text)) {
        warning(this, "SPARTA-GUI Warning", "Cannot open file " + fileName + ":",
                file.errorString());
    } else {
        file.close();
        auto *viewer = new FileViewer(fileName, this);
        viewer->show();
    }
}

// open one or more image or movie files in a standalone snapshot viewer
void SpartaGui::openImages()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, "Open Image or Movie File(s)", currentDir,
        "Image and movie files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm *.gif *.tif *.tiff *.tga "
        "*.eps *.sgi *.webp *.mp4 *.m4v *.mkv *.webm *.avi *.mov *.mpg *.mpeg *.ogv *.wmv "
        "*.flv);;Image files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm *.gif *.tif *.tiff *.tga *.eps "
        "*.sgi *.webp);;Movie files (*.mp4 *.m4v *.mkv *.webm *.avi *.mov *.mpg *.mpeg *.ogv *.wmv "
        "*.flv *.gif);;All files (*)");
    if (files.isEmpty()) return;

    auto *win = ViewerWindow::forSequence(files.first());
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->show();

    // the import dialog of a movie file is modal to the (already visible)
    // slide show window, so a movie must not be added before it is shown
    auto *viewer = win->sequence();
    for (const QString &f : files) {
        if (isMovieFile(f))
            viewer->addMovie(f);
        else
            viewer->addImage(f);
    }

    // every movie import was canceled or failed and no image was selected
    if (viewer->imageCount() == 0) win->close();
}

void SpartaGui::purgeInspectList()
{
    // iterator loop: erase() both removes the entry (a range-for would be left
    // with invalidated iterators) and hands back the next valid position
    for (auto it = inspectList.begin(); it != inspectList.end();) {
        auto *item = *it;
        if (item->info && !item->info->isVisible()) {
            delete item->info;
            item->info = nullptr;
        }
        if (item->image && !item->image->isVisible()) {
            delete item->image;
            item->image = nullptr;
        }
        if (!item->image && !item->info) {
            delete item;
            it = inspectList.erase(it);
        } else {
            ++it;
        }
    }
}

// read restart file into SPARTA instance and launch image viewer
void SpartaGui::inspectFile(const QString &fileName)
{
    // empty name means the file dialog was cancelled: nothing to inspect
    if (fileName.isEmpty()) return;

    QFile file(fileName);
    auto shortName = QFileInfo(fileName).fileName();

    purgeInspectList();
    auto *ilist  = new InspectData;
    ilist->info  = nullptr;
    ilist->image = nullptr;
    inspectList.append(ilist);

    if (file.size() > Cfg::INSPECT_WARN_SIZE) {
        QMessageBox mb;
        mb.setWindowTitle("  Warning:  Large Restart File  ");
        mb.setWindowIcon(windowIcon());
        mb.setText(QString("<center>The restart file ") + shortName + " is large</center>");
        QString details = "Inspecting the restart file %1 with SPARTA-GUI may need an additional "
                          "%2 GB of free RAM (or more) to proceed";
        mb.setDetailedText(details.arg(shortName).arg(file.size() / Cfg::INSPECT_GB_PER_BYTE));
        mb.setInformativeText("Do you want to continue?");
        mb.setIconPixmap(
            QIcon(":/icons/warning.svg").pixmap(QSize(64, 64), mb.devicePixelRatioF()));
        mb.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        mb.setDefaultButton(QMessageBox::No);
        mb.setEscapeButton(QMessageBox::No);
        mb.setFont(font());

        auto *button = mb.button(QMessageBox::Yes);
        button->setIcon(QIcon(":/icons/dialog-ok.svg"));
        button = mb.button(QMessageBox::No);
        button->setIcon(QIcon(":/icons/dialog-no.svg"));

        int rv = mb.exec();
        switch (rv) {
            case QMessageBox::No:
                return;
            case QMessageBox::Yes: // fallthrough
            default:
                // do nothing
                break;
        }
    }

    if (!file.open(QIODevice::ReadOnly)) {
        warning(this, "SPARTA-GUI Warning", "Cannot open file " + fileName + ":",
                file.errorString());
        return;
    }
    file.close();

    if (!isRestartFile(fileName)) {
        warning(this, "SPARTA-GUI Warning", "File " + fileName + " is not a SPARTA restart file.");
        return;
    }

    // SPARTA is not re-entrant, so we can only query SPARTA when it is not running a simulation
    if (!sparta.isRunning()) {
        startSparta();
        // SPARTA has no "info" command, so we capture the screen output of the
        // read_restart command itself. It summarizes the restored simulation:
        // grid cells, surface elements, particles, and species/mixture counts.
        capturer->beginCapture();
        sparta.command("clear");
        clearVariables();
        sparta.command(QString("read_restart %1").arg(fileName));
        capturer->endCapture();
        auto info = capturer->getCapture();

        const QString errmsg = sparta.lastErrorMessage();
        if (!errmsg.isEmpty() && !errmsg.contains("Invalid SPARTA handle")) {
            warning(this, "SPARTA-GUI Warning",
                    "Error reading restart file " + fileName + ":", errmsg);
            return;
        }

        // read_restart does not restore the RNG seed (it is a runtime command, not
        // stored in the restart file). Rendering the restored state runs "run 0",
        // which requires a seeded RNG whenever the restart defines a collide or
        // react style. Provide a fixed seed here so restart inspection can always
        // create an image; the value is irrelevant for a static visualization.
        sparta.command("seed 12345");

        auto infolog = QString("%1.info.log").arg(fileName);
        QFile dumpinfo(infolog);
        if (dumpinfo.open(QIODevice::WriteOnly)) {
            dumpinfo.write(info.c_str(), info.size());
            dumpinfo.close();
            auto *infoviewer = new FileViewer(
                infolog, this, QString("SPARTA-GUI: restart info for %1").arg(shortName));
            infoviewer->show();
            ilist->info = infoviewer;
            dumpinfo.remove();
            auto *inspect_image = ViewerWindow::forSnapshot(fileName, &sparta, this, this);
            inspect_image->setFont(font());
            inspect_image->show();
            ilist->image = inspect_image;
        }
    }
}

// write file and update CWD to its folder

void SpartaGui::writeFile(const QString &fileName)
{
    // empty name means the save dialog was cancelled: nothing to save to
    if (fileName.isEmpty()) return;

    QFileInfo path(fileName);
    QFile file(path.absoluteFilePath());

    if (!file.open(QIODevice::WriteOnly | QFile::Text)) {
        warning(this, "SPARTA-GUI Warning", "Cannot save to file " + fileName + ":",
                file.errorString());
        return;
    }
    // update the session state only after the file was opened successfully
    currentFile = path.fileName();
    currentDir  = path.absolutePath();
    setWindowTitle(QString("SPARTA-GUI - Editor - " + currentFile));
    QDir::setCurrent(currentDir);

    updateRecents(path.absoluteFilePath());

    QTextStream out(&file);
    QString text = textEdit->toPlainText();
    out << text;
    if (!text.endsWith('\n')) out << "\n"; // add final newline if missing
    file.close();
    dirstatus->setText(QString(" Directory: ") + currentDir);
    // update list of files for completion since we may have changed the working directory
    textEdit->setFileList();
    textEdit->document()->setModified(false);
    // the buffer now matches the file on disk; no recovery copy needed
    clearRecoveryFile();
}

void SpartaGui::save()
{
    purgeInspectList();
    QString fileName = currentFile;
    // If we don't have a filename from before, get one.
    if (fileName.isEmpty()) fileName = QFileDialog::getSaveFileName(this, "Save");

    writeFile(fileName);
}

void SpartaGui::saveAs()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as");
    writeFile(fileName);
}

void SpartaGui::quit()
{
    stopAndReapRunner();

    autoSave();
    if (textEdit->document()->isModified()) {
        int rv = showUnsavedChangesDialog(this, currentFile,
                                          "Do you want to save the file before exiting?");
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::Cancel:
                return;
            case QMessageBox::No: // fallthrough
            default:
                // do nothing
                break;
        }
    }

    // store some global settings
    QSettings settings;
    if (!isMaximized()) {
        settings.setValue(Keys::MAINX, width());
        settings.setValue(Keys::MAINY, height());
    }
    // persist session state for the next launch and drop the crash-recovery
    // file (a clean exit needs no recovery)
    settings.setValue(Keys::WINGEOMETRY, saveGeometry());
    settings.setValue(Keys::LAST_FILE, currentFile.isEmpty()
                                           ? QString()
                                           : QDir(currentDir).absoluteFilePath(currentFile));
    clearRecoveryFile();
    panels->saveLayout(settings);
    settings.sync();

#if QT_CONFIG(clipboard)
    if (auto *clip = QGuiApplication::clipboard()) clip->clear();
#endif

    // tear down SPARTA-GUI and close / finalize SPARTA instance

    removeEventFilter(this);
    {
        StdoutSilencer guard;
        sparta.finalize();
    }
    spartastatus->hide();

    // quit application
    QCoreApplication::quit();
}

void SpartaGui::copy()
{
#if QT_CONFIG(clipboard)
    textEdit->copy();
#endif
}

void SpartaGui::cut()
{
#if QT_CONFIG(clipboard)
    textEdit->cut();
#endif
}

void SpartaGui::paste()
{
#if QT_CONFIG(clipboard)
    textEdit->paste();
#endif
}

void SpartaGui::undo()
{
    textEdit->undo();
}

void SpartaGui::redo()
{
    textEdit->redo();
}

void SpartaGui::syncRunControls()
{
    const bool running = sparta.isRunning() || workerActive();
    // Nothing can be run at all until there is a library.  The setup card above
    // the editor says why and offers the two ways to get one; these controls
    // being greyed is what makes the card's claim visibly true rather than an
    // assertion the user has to take on faith.
    const bool canRun = sparta.hasLibrary() && !running;
    if (stopAction) stopAction->setEnabled(running);
    // Run and Create Image refuse with a modal while something is running;
    // greyed out beats a dialog that says no.
    if (runAction) runAction->setEnabled(canRun);
    if (imageAction) imageAction->setEnabled(canRun);
    // Extending needs a state to continue from and nothing in flight.  Greyed
    // out rather than hidden, so the entry is discoverable before the first run
    // explains what it is for.
    if (extendAction) extendAction->setEnabled(canRun && hasSystemState());
    if (restartAction) restartAction->setEnabled(canRun && hasSystemState());
}

void SpartaGui::stopRun()
{
    if (!sparta.isRunning()) return;
    sparta.forceTimeout();
}

bool SpartaGui::workerActive() const
{
    return runner && runner->isRunning();
}

void SpartaGui::stopAndReapRunner()
{
    // Guarded on thread liveness rather than on sparta.isRunning(), which was
    // the wrong question in both directions.
    //
    // It is false while the worker thread is still working: runflag is set only
    // inside SPARTA's run loop, so through read_surf, create_grid,
    // create_particles and the rest of a deck's setup this block was skipped
    // and the instance was closed, or replaced, underneath a live thread.
    //
    // And it stayed true after an error until Run::command was fixed to reset
    // it -- at which point the thread had long since finished and been freed by
    // its own finished() connection, so wait() ran on freed memory.
    //
    // runner is null once that connection has fired, so a stale pointer cannot
    // be reached through here at all.
    if (!runner) return;
    if (runner->isRunning()) {
        stopRun();
        runner->wait();
    }
    // Freeing it is the finished() handler's job, and only its job.  Every
    // runner has one, it runs exactly once, and doing it here as well raced
    // with it: this call could delete the object before the queued handler had
    // run, or -- if a later run had already installed a new runner -- leave the
    // old one for a handler that no longer recognised it.  Dropping the
    // reference is all that is wanted here.  The runner is a child of this
    // window in any case, so it cannot outlive it.
    runner = nullptr;
}

void SpartaGui::logUpdate()
{
    progress->setValue(updateRunStatus());

    if (logwindow) {
        const auto text = capturer->getChunk();
        if (!text.empty()) {
            logwindow->moveCursor(QTextCursor::End);
            logwindow->insertPlainText(text.c_str());
            logwindow->moveCursor(QTextCursor::End);
        }
    }

    // Everything read out of the stats cache comes out under one lock.
    //
    // The timestep was read here, outside the lock, and the column values below
    // it, inside -- so a run that completed another stats line in between gave
    // this one step's x with the next step's y, and the chart quietly plotted
    // points that never existed.  The "setup" flag had the same problem, and
    // reading `bigint` is a build constant that need not be in the critical
    // section at all.
    const bool bigint4 = sparta.extractSetting("bigint") == 4;

    int step = 0;
    int ncols = 0;
    sparta.lastThermo("lock", 0);
    // thermo data is not yet valid during setup
    const bool insetup = sparta.lastThermoAs<int>("setup", 0) != 0;
    if (!insetup) {
        step = bigint4 ? sparta.lastThermoAs<int>("step", 0)
                       : static_cast<int>(sparta.lastThermoAs<int64_t>("step", 0));
        if (chartwindow && sparta.isRunning()) ncols = sparta.lastThermoAs<int>("num", 0);
        if (ncols > 0) updateChartData(step, ncols);
    }
    sparta.lastThermo("unlock", 0);

    // Not skipped during setup any more.  This used to sit behind an early
    // return taken while the run was setting up, so any image the deck wrote
    // before the first timestep never reached the slide show.
    updateSlideShow();

    // let a parametric sweep sample the current thermo values each tick
    emit thermoSampled();
}

int SpartaGui::updateRunStatus()
{
    if (!sparta.isRunning()) return 1000;

    // estimate completion percentage
    double t_elapsed = sparta.getThermo("cpu");
    double t_remain  = sparta.getThermo("cpuremain");
    double t_total   = t_elapsed + t_remain + 1.0e-10;
    int completed    = t_elapsed / t_total * 1000.0;
    // update cpu usage
    int percent_cpu = static_cast<int>(sparta.getThermo("cpuuse"));
    // clear any pending error messages from polling those thermo keywords
    (void)sparta.lastErrorMessage(); // read-and-clear any pending error

    cpuuse->setText(QString("%1%CPU").arg(percent_cpu, 4));
    // pick a color bucket for the CPU-usage label. Re-applying a stylesheet
    // forces an expensive Qt style re-parse/polish, and this runs on every
    // poll tick (~100 Hz) during a run, so only restyle when the bucket
    // actually changes rather than every tick.
    int bucket; // 0=black 1=darkblue 2=firebrick 3=gold 4=forestgreen
    if (percent_cpu < 25.0 * nthreads)
        bucket = 0;
    else if (percent_cpu < 50.0 * nthreads)
        bucket = 1;
    else if (percent_cpu > 100.0 * nthreads + 50.0)
        bucket = 2;
    else if (percent_cpu < 100.0 * nthreads - 50.0)
        bucket = 2;
    else if (percent_cpu > 100.0 * nthreads + 20.0)
        bucket = 3;
    else if (percent_cpu < 100.0 * nthreads - 20.0)
        bucket = 3;
    else
        bucket = 4;
    if (bucket != lastCpuBucket) {
        lastCpuBucket = bucket;
        switch (bucket) {
            case 0:
                cpuuse->setStyleSheet("QLabel {background-color: black; color: white;}");
                break;
            case 1:
                cpuuse->setStyleSheet("QLabel {background-color: darkblue; color: white;}");
                break;
            case 2:
                cpuuse->setStyleSheet("QLabel {background-color: firebrick; color: white;}");
                break;
            case 3:
                cpuuse->setStyleSheet("QLabel {background-color: gold; color: black;}");
                break;
            default:
                cpuuse->setStyleSheet("QLabel {background-color: forestgreen; color: white;}");
                break;
        }
    }

    // 1-based input line -> 0-based editor block (see runDone())
    void *ptr = sparta.lastThermo("line", 0);
    if (ptr) {
        const int ln = *static_cast<int *>(ptr);
        if (ln >= 1) textEdit->setHighlight(ln - 1, false);
    }

    if (varwindow) {
        int nvar = sparta.idCount("variable");
        QString varinfo("\n");
        for (int i = 0; i < nvar; ++i)
            varinfo += sparta.variableInfo(i);
        if (nvar == 0) varinfo += "  (none)  ";

        varwindow->setText(varinfo);
        varwindow->adjustSize();
    }
    return completed;
}

void SpartaGui::updateChartData(int step, int ncols)
{
    // check if the column assignment has changed
    // if yes, delete charts and start over
    if (chartwindow->numCharts() > 0) {
        int count     = 0;
        bool do_reset = false;
        if (step < chartwindow->getStep()) do_reset = true;
        for (int i = 0, idx = 0; i < ncols; ++i) {
            QString label = sparta.lastThermoString("keyword", i);
            // no need to store the timestep column
            if (label == "Step") continue;
            if (!chartwindow->hasTitle(label, idx)) {
                do_reset = true;
            } else {
                ++count;
            }
            ++idx;
        }
        if (chartwindow->numCharts() != count) do_reset = true;
        if (do_reset) chartwindow->resetCharts();
    }

    if (chartwindow->numCharts() == 0) {
        for (int i = 0; i < ncols; ++i) {
            QString label = sparta.lastThermoString("keyword", i);
            // no need to store the timestep column
            if (label == "Step") continue;
            chartwindow->addChart(label, i);
        }
    }

    for (int i = 0; i < ncols; ++i) {
        const int datatype = sparta.lastThermoAs<int>("type", i);
        chartwindow->addData(step, lastThermoData(sparta, datatype, i), i);
    }
}

// Create the viewer panel on first use and install it in its dock. The
// individual sources are added as they are first needed -- a deck that never
// renders anything never builds an image viewer.
void SpartaGui::ensureViewerPanel()
{
    if (viewer) return;

    viewer = new ViewerPanel;
    panels->setPanelWidget(PanelManager::Viewer, viewer, viewer->title());
    connect(viewer, &ViewerPanel::titleChanged, this,
            [this](const QString &title) { panels->setPanelTitle(PanelManager::Viewer, title); });
    connect(viewer, &ViewerPanel::closeRequested, this,
            [this]() { panels->closePanel(PanelManager::Viewer); });
    connect(viewer, &ViewerPanel::sourceChanged, this,
            [](int which) { QSettings().setValue(Keys::VIEWERSOURCE, which); });

#if defined(SPARTA_GUI_HAVE_VTK)
    viewer->addSource(ViewerPanel::Scene, new VtkScene);
#endif
}

void SpartaGui::updateSlideShow()
{
    // Under the lock, like every other read of the cache.  This one was not,
    // and it is the read that matters most: last_thermo("imagename") hands back
    // the internal std::string's buffer, and DumpImage::write() assigns to that
    // same string from the worker thread every time it finishes a frame.  A
    // deck writing images during a run therefore had the GUI copying from a
    // buffer that was being reallocated underneath it -- a garbled file name
    // when it was lucky, freed memory when it was not.
    sparta.lastThermo("lock", 0);
    const QString imagefile = sparta.lastThermoString("imagename", 0);
    sparta.lastThermo("unlock", 0);
    if (imagefile.isEmpty()) return;

    ensureViewerPanel();
    if (!viewer->sequence())
        viewer->addSource(ViewerPanel::Sequence, new SlideShow(currentFile, this));

    viewer->sequence()->addImage(imagefile);

    // Same again: frames pile up in the sequence either way, but the panel only
    // comes forward in a workspace that shows pictures.
    if (QSettings().value(Keys::VIEWSLIDE, true).toBool() &&
        PanelManager::modeShows(panels->currentMode(), PanelManager::Viewer)) {
        panels->openPanel(PanelManager::Viewer);
        // frames arriving on their own must not take the view away from
        // whatever the user chose to look at during the run
        viewer->showSource(ViewerPanel::Sequence);
    }
}

void SpartaGui::modified()
{
    const QString modflag(" - *modified*");
    auto title = windowTitle().remove(modflag);
    if (textEdit->document()->isModified()) {
        textEdit->setStyleSheet("");
        setWindowTitle(title + modflag);
    } else
        setWindowTitle(title);
}

void SpartaGui::warnHighBufferUsage()
{
    // check stdout capture buffer utilization and print warning message if large

    double bufferuse = capturer->getBufferUse();
    if (bufferuse > Cfg::BUFFER_WARNING_THRESHOLD) {
        int thermo_val = sparta.extractSetting("stats_every");
        int thermo_suggest =
            Cfg::THERMO_SUGGEST_MULTIPLIER * static_cast<int>(round(bufferuse * thermo_val));
        int update_val =
            QSettings().value(Keys::UPDFREQ, Cfg::DATA_UPDATE_INTERVAL_DEFAULT).toInt();
        int update_suggest = std::max(1, update_val / 5);

        QString mesg1("<p align=\"justify\">The I/O buffer for capturing the SPARTA screen "
                      "output was used by up to %1%.</p>"
                      "<p align=\"justify\"><b>This can slow down the simulation.</b></p>");
        QString mesg2("<p align=\"justify\">Please consider reducing the amount of output "
                      "to the screen, for example by increasing the stats interval in the "
                      "input from %1 to %2, or reducing the data update interval in the "
                      "preferences from %3 to %4, or something similar.</p>");

        critical(this, "SPARTA-GUI Warning: High I/O Buffer Usage",
                 mesg1.arg(static_cast<int>(100.0 * bufferuse)),
                 mesg2.arg(thermo_val).arg(thermo_suggest).arg(update_val).arg(update_suggest));
    }
}

void SpartaGui::finalizeChartData()
{
    if (chartwindow) {
        int step = 0;
        if (sparta.extractSetting("bigint") == 4)
            step = sparta.lastThermoAs<int>("step", 0);
        else
            step = static_cast<int>(sparta.lastThermoAs<int64_t>("step", 0));
        const int ncols = sparta.lastThermoAs<int>("num", 0);
        // decide once before the loop: testing numCharts() per column would stop
        // creating charts as soon as the first addChart() call succeeded
        const bool needcharts = (chartwindow->numCharts() == 0);
        for (int i = 0; i < ncols; ++i) {
            if (needcharts) {
                QString label = sparta.lastThermoString("keyword", i);
                // no need to store the timestep column
                if (label == "Step") continue;
                chartwindow->addChart(label, i);
            }
            const int datatype = sparta.lastThermoAs<int>("type", i);
            chartwindow->addData(step, lastThermoData(sparta, datatype, i), i);
        }
        chartwindow->resetZoom();
        chartwindow->setRangeEnabled(true);
    }
}

void SpartaGui::runDone()
{
    if (logupdater) {
        logupdater->stop();
        delete logupdater;
        logupdater = nullptr;
    }
    syncRunControls();
    progress->setValue(Cfg::PROGRESS_MAXIMUM);
    textEdit->setHighlight(CodeEditor::NO_HIGHLIGHT, false);

    // The chart was built before the run started, so the units it read were the
    // ones in force then, not any the deck went on to set.  Now that the thread
    // has finished, unit_style is nobody else's to free and can be read safely.
    if (chartwindow) {
        const auto *unitptr = static_cast<const char *>(sparta.extractGlobal("units"));
        if (unitptr) chartwindow->setUnits(QString::fromUtf8(unitptr));
    }

    capturer->endCapture();

    if (logwindow) {
        auto log = capturer->getCapture();
        logwindow->insertPlainText(log.c_str());
        logwindow->moveCursor(QTextCursor::End);
    }

    warnHighBufferUsage();

    finalizeChartData();

#if defined(SPARTA_GUI_HAVE_VTK)
    // A finished run is the state worth looking at, and it is the only point at
    // which particles exist. Refresh only where the pictures are what the window
    // is showing: rendering three dump files costs a "run 0" apiece, which is
    // not something to spend on a workspace the user is not looking at.
    if (PanelManager::modeShows(panels->currentMode(), PanelManager::Viewer))
        refreshDocked3DScene();
#endif

    bool success         = true;
    bool valid           = true;
    const QString errmsg = sparta.lastErrorMessage();

    if (!errmsg.isEmpty()) {
        // ignore "Invalid SPARTA handle", but report other errors
        if (!errmsg.contains("Invalid SPARTA handle")) {
            success = false;
        } else {
            valid = false;
        }
    }

    // last_thermo("line") is the 1-based input-script line of the failing
    // command; the editor's setHighlight()/setCursor() take a 0-based block
    // index (as the diagnostics list does with line-1), so convert here.
    int nline = CodeEditor::NO_HIGHLIGHT;
    if (valid) {
        void *ptr = sparta.lastThermo("line", 0);
        if (ptr) {
            const int ln = *static_cast<int *>(ptr);
            if (ln >= 1) nline = ln - 1;
        }
    }

    if (success) {
        status->setText(Cfg::STATUS_READY);
        cpuuse->setText(Cfg::STATUS_ZERO_CPU);
    } else {
        status->setText("Failed.");
        textEdit->setHighlight(nline, true);
        critical(this, "SPARTA-GUI Error", "<p>Error running SPARTA:</p>",
                 QString("<p><pre>%1</pre></p>").arg(errmsg));
    }
    textEdit->setCursor(nline);
    textEdit->setFileList();
    progress->hide();
    cpuuse->hide();
    dirstatus->show();

    // archive this run for provenance if the user opted in (default off)
    archiveFinishedRun(success);

    // let a parametric sweep (or any observer) advance to the next run
    emit runFinished(success);
}

void SpartaGui::restartSparta()
{
    // workerActive() too: runflag is clear while the thread is still in a
    // deck's setup, and a second run started there overwrote the runner and
    // issued "clear" -- tearing SPARTA down under the first one.
    if (sparta.isRunning() || workerActive()) {
        warning(this, "SPARTA-GUI Warning", "Must stop current run before relaunching SPARTA");
        return;
    }
    {
        StdoutSilencer guard;
        sparta.close();
    }
}

void SpartaGui::ensureLogPanel()
{
    if (logwindow) return;
    logwindow = new LogWindow(currentFile, this);
    logwindow->setReadOnly(true);
    logwindow->setCenterOnScroll(false);
    logwindow->setLineWrapMode(LogWindow::NoWrap);
    // Plain "Output": there is no run behind it yet, so the "Output - <file> -
    // Run <n>" title createLogWindow() uses would be claiming one.
    panels->setPanelWidget(PanelManager::Log, logwindow, "Output");
}

void SpartaGui::ensureChartPanel()
{
    if (chartwindow) return;
    chartwindow = new ChartWindow(currentFile, this);
    // Plain "Charts", for the same reason ensureLogPanel() uses plain "Output":
    // the "Charts - <file> - Run <n>" title createChartWindow() applies would be
    // naming a run that has not happened.
    panels->setPanelWidget(PanelManager::Chart, chartwindow, "Charts");
    chartwindow->setNorm(false);
    chartwindow->setRangeEnabled(false);
}

void SpartaGui::createLogWindow(QSettings &settings)
{
    logwindow = new LogWindow(currentFile, this);
    logwindow->setReadOnly(true);
    // Deliberately NOT setCenterOnScroll(true). That parks the last line in the
    // middle of the viewport, so half the panel sits empty below the newest
    // output and the same number of earlier lines is pushed off the top -- on a
    // tall panel that is a lot of log the user cannot see. Scrolled normally,
    // the last line sits at the bottom and the whole panel shows text.
    logwindow->setCenterOnScroll(false);
    logwindow->moveCursor(QTextCursor::End);
    logwindow->setLineWrapMode(LogWindow::NoWrap);

    const bool keepOld = !settings.value(Keys::LOGREPLACE, true).toBool();
    panels->setPanelWidget(PanelManager::Log, logwindow,
                          QString("Output - %1 - Run %2").arg(currentFile).arg(runCounter),
                          keepOld);

    // Only where the workspace has room for it, same rule as the chart below.
    // Starting a run from Analyze or Visualize is a request to watch the plots
    // or the pictures; the console output taking a column of those workspaces
    // is exactly what choosing them said not to do. The output is still being
    // collected, and the Run workspace (or the View menu) still has it.
    if (settings.value(Keys::VIEWLOG, true).toBool() &&
        PanelManager::modeShows(panels->currentMode(), PanelManager::Log))
        panels->openPanel(PanelManager::Log);
    else
        panels->closePanel(PanelManager::Log);
}

void SpartaGui::createChartWindow(QSettings &settings)
{
    chartwindow = new ChartWindow(currentFile, this);

    const bool keepOld = !settings.value(Keys::CHARTREPLACE, true).toBool();
    panels->setPanelWidget(PanelManager::Chart, chartwindow,
                          QString("Charts - %1 - Run %2").arg(currentFile).arg(runCounter),
                          keepOld);

    const auto *unitptr = static_cast<const char *>(sparta.extractGlobal("units"));
    if (unitptr) chartwindow->setUnits(QString::fromUtf8(unitptr));
    // SPARTA stats output has no equivalent of LAMMPS' per-atom normalization
    chartwindow->setNorm(false);
    chartwindow->setRangeEnabled(false);

    // Only where the workspace has room for it. The editing and running
    // workspaces are a deck beside its output, deliberately, so a run must not
    // push a chart into that column and halve both -- the plots are what the
    // Analyze workspace is for, one keystroke away.
    if (settings.value(Keys::VIEWCHART, true).toBool() &&
        PanelManager::modeShows(panels->currentMode(), PanelManager::Chart))
        panels->openPanel(PanelManager::Chart);
    else
        panels->closePanel(PanelManager::Chart);
}

void SpartaGui::doRun(bool use_buffer)
{
    // workerActive() too: runflag is clear while the thread is still in a
    // deck's setup, and a second run started there overwrote the runner and
    // issued "clear" -- tearing SPARTA down under the first one.
    if (sparta.isRunning() || workerActive()) {
        warning(this, "SPARTA-GUI Warning", "Must stop current run before starting a new run");
        return;
    }

    // a run operates on the editor buffer, so make sure it is the visible page
    showEditor();

    // Starting a run is a change of task, so bring up the Run workspace -- but
    // only for the first run of a session. Doing it on every run would fight a
    // user who deliberately switched to Analyze to watch the charts.
    if (!ranThisSession) {
        ranThisSession = true;
        QSettings settings;
        if (settings.value(Keys::RUNMODE_AUTOSWITCH, true).toBool())
            panels->applyMode(PanelManager::RunMode);
    }

    purgeInspectList();
    autoSave();

    // the SPARTA "quit" command calls exit() and thus would terminate not just
    // the run but the entire SPARTA-GUI process. Warn before running such input.
    if (textEdit->toPlainText().contains(
            QRegularExpression(QStringLiteral(R"(^\s*quit\b)"), // clazy:exclude=use-static-qregularexpression
                               QRegularExpression::MultilineOption))) {
        QMessageBox msg(QMessageBox::Warning, "SPARTA-GUI Warning",
                        "The input contains a 'quit' command.\n\n"
                        "Executing 'quit' will terminate not only the SPARTA run "
                        "but the entire SPARTA-GUI application.\n\n"
                        "Do you want to run the input anyway?",
                        QMessageBox::Yes | QMessageBox::No, this);
        msg.setDefaultButton(QMessageBox::No);
        if (msg.exec() != QMessageBox::Yes) return;
    }

    if (!use_buffer && textEdit->document()->isModified()) {
        int rv = showUnsavedChangesDialog(this, currentFile,
                                          "Do you want to save the buffer before running SPARTA?");
        switch (rv) {
            case QMessageBox::Yes:
                save();
                break;
            case QMessageBox::No:
                break;
            case QMessageBox::Cancel: // fallthrough
            default:
                return;
        }
    }

    QSettings settings;
    progress->setValue(0);
    dirstatus->hide();
    progress->show();
    cpuuse->show();
    lastCpuBucket = -1; // force the cpuuse stylesheet to be applied on the first poll

    int numthreads = nthreads;
    int accel      = settings.value(Keys::ACCELERATOR, AcceleratorTab::None).toInt();
    if (accel != AcceleratorTab::Kokkos) numthreads = 1;
    if (numthreads > 1)
        status->setText(QString("Running SPARTA with %1 thread(s)...").arg(numthreads));
    else
        status->setText(QString("Running SPARTA ..."));
    status->repaint();
    startSparta();
    if (!sparta.isOpen()) return;
    capturer->beginCapture();

    ++runCounter;

    // must delete all variables since clear does not delete them
    clearVariables();

    // define "gui_run" variable set to runCounter value
    sparta.command(QString("variable gui_run index %1").arg(runCounter));

    // re-create index variables from the Set Variables dialog so they
    // override definitions in the input, like -var does on the command line
    for (const auto &var : std::as_const(variables)) {
        if (!var.first.isEmpty() && !var.second.isEmpty())
            sparta.command(QString("variable %1 index %2").arg(var.first, var.second));
    }
    // apply https proxy setting: prefer environment variable or fall back to preferences value
    applyProxySetting(sparta, settings);

    if (use_buffer) {
        // always add final newline since the text edit widget does not do it
        launchRunner((textEdit->toPlainText() + "\n").toStdString(), {}, true);
    } else {
        launchRunner({}, currentFile.toStdString(), true);
    }

    if (viewer && viewer->sequence()) {
        viewer->unlockSource();
        viewer->sequence()->clear();
        panels->closePanel(PanelManager::Viewer);
    }
    syncRunControls();
}

/* ---------------------------------------------------------------------- */

void SpartaGui::launchRunner(std::string input, std::string file, bool clearfirst)
{
    QSettings settings;
    runner = new SpartaRunner(this);
    runner->setupRun(&sparta, std::move(input), std::move(file), clearfirst);

    connect(runner, &SpartaRunner::resultReady, this, &SpartaGui::runDone);
    // Clear the member as well as freeing the object.  Connecting finished
    // straight to deleteLater left `runner` pointing at freed memory the moment
    // a run ended, and every use of it after that was live only by luck of the
    // guard in front of it.  The captured pointer is compared before clearing,
    // so a runner created for a later run is never disowned by an older one's
    // finished().
    connect(runner, &SpartaRunner::finished, this, [this, r = runner]() {
        // Drop the reference only if it is still this runner -- a later run may
        // have installed its own by the time this arrives -- but free the object
        // unconditionally, because this handler is the one place that does.
        if (runner == r) runner = nullptr;
        r->deleteLater();
        // The run controls are also synced from runDone(), but that runs off
        // resultReady() while this thread is still alive -- so anything gated on
        // workerActive(), such as Extend Run and Write Restart, was switched off
        // there and had nothing to switch it back on.  This is the point where
        // the worker is really gone.
        syncRunControls();
    });
    // Built before the thread is let loose, not after.  createChartWindow()
    // reads extractGlobal("units"), which hands back update->unit_style itself
    // -- and Update::set_units() does `delete [] unit_style` before replacing
    // it.  A deck whose first lines include a `units` command (the normal place
    // for it) had the worker thread freeing that buffer microseconds after
    // start(), while this thread was copying out of it.
    createLogWindow(settings);

    createChartWindow(settings);

    runner->start();

    logupdater = new QTimer(this);
    connect(logupdater, &QTimer::timeout, this, &SpartaGui::logUpdate);
    logupdater->start(settings.value(Keys::UPDFREQ, Cfg::DATA_UPDATE_INTERVAL_DEFAULT).toInt());
}

/* ---------------------------------------------------------------------- */

bool SpartaGui::hasSystemState()
{
    // A box is the minimum a continued run needs: `run` without one fails, and
    // so does writing a restart.  isRunning()/workerActive() because SPARTA is
    // not re-entrant -- commands may only be issued between runs.
    return sparta.isOpen() && !sparta.isRunning() && !workerActive() &&
           sparta.extractSetting("box_exist") != 0;
}

/* ---------------------------------------------------------------------- */

void SpartaGui::extendRun()
{
    if (sparta.isRunning() || workerActive()) {
        warning(this, "SPARTA-GUI Warning", "Must stop the current run before extending it");
        return;
    }
    if (!hasSystemState()) {
        warning(this, "SPARTA-GUI Warning",
                "Cannot extend a run without a system state.\n"
                "Run the input at least as far as defining the box and grid first.");
        return;
    }

    bool ok           = false;
    const int nsteps  = QInputDialog::getInt(this, "Extend Run", "Number of steps to add:",
                                             extendSteps, 1, std::numeric_limits<int>::max(), 1, &ok);
    if (!ok) return;
    extendSteps = nsteps;

    QSettings settings;
    progress->setValue(0);
    progress->show();
    status->setText(QString("Extending run by %1 steps ...").arg(nsteps));
    status->repaint();

    capturer->beginCapture();

    // The windows of the run being extended are reused; they are only created
    // when missing, which happens when the state came from an inspected restart
    // file rather than from a run in this session.
    if (!logwindow) createLogWindow(settings);
    if (!chartwindow) createChartWindow(settings);

    logwindow->moveCursor(QTextCursor::End);
    logwindow->insertPlainText(
        QString("\n========== Extending run by %1 steps ==========\n\n").arg(nsteps));
    logwindow->moveCursor(QTextCursor::End);

    // "pre yes", not "pre no", even though the state is already set up: each run
    // gets a fresh runner thread with its own thread pool, and only the setup
    // phase re-initialises the per-thread data of a threaded accelerator for
    // that pool -- skipping it crashes a Kokkos/OpenMP run.  "post no" only
    // drops the timing summary of the extension.  SPARTA clears a forced
    // timeout at the top of Run::command, so a run stopped with the Stop button
    // does not need it undone here.
    //
    // clearfirst is false: "clear" would destroy the very state being extended.
    launchRunner(QString("run %1 pre yes post no\n").arg(nsteps).toStdString(), {}, false);
    syncRunControls();
}

/* ---------------------------------------------------------------------- */

void SpartaGui::writeRestart()
{
    // SPARTA is not re-entrant: commands may only be issued between runs.
    if (sparta.isRunning() || workerActive()) {
        warning(this, "SPARTA-GUI Warning",
                "Must stop the current run before writing a restart file");
        return;
    }
    if (!hasSystemState()) {
        warning(this, "SPARTA-GUI Warning",
                "Cannot write a restart file without a system state.\n"
                "Run the input at least as far as defining the box and grid first.");
        return;
    }

    QFileInfo deck(currentFile);
    const QString suggested =
        QDir::current().absoluteFilePath((deck.completeBaseName().isEmpty()
                                              ? QStringLiteral("sparta")
                                              : deck.completeBaseName()) +
                                         ".restart");
    QString fileName = QFileDialog::getSaveFileName(this, "Write Restart File", suggested,
                                                    "Restart files (*.restart);;All files (*)");
    if (fileName.isEmpty()) return;
    if (!fileName.endsWith(".restart")) fileName += ".restart";

    {
        StdoutSilencer guard;
        // quoted: a path chosen from a file dialog may contain spaces, and
        // SPARTA splits its command line on them
        sparta.command(QString("write_restart '%1'").arg(fileName));
    }

    const QString errmsg = sparta.lastErrorMessage();
    if (!errmsg.isEmpty()) {
        critical(this, "SPARTA-GUI Error", "Error writing restart file:", errmsg);
    } else {
        status->setText(QString("Wrote restart file %1").arg(QFileInfo(fileName).fileName()));
    }
}

/* ---------------------------------------------------------------------- */

void SpartaGui::ensureSweepPanel()
{
    if (sweepPanel) return;
    sweepPanel = new SweepPanel(this, this, &sparta);
    panels->setPanelWidget(PanelManager::Sweep, sweepPanel, "Parameter Sweep");
}

void SpartaGui::runSweep()
{
    ensureSweepPanel();
    panels->openPanel(PanelManager::Sweep);
}

QString SpartaGui::recoveryFilePath() const
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) +
           "/recovery/session.in";
}

void SpartaGui::startRecoveryTimer()
{
    const int secs = QSettings().value(Keys::AUTOSAVE_INTERVAL, Cfg::RECOVERY_INTERVAL_DEFAULT).toInt();
    if (secs <= 0) return; // disabled
    if (!recoveryTimer) {
        recoveryTimer = new QTimer(this);
        connect(recoveryTimer, &QTimer::timeout, this, &SpartaGui::writeRecoveryFile);
    }
    recoveryTimer->start(secs * 1000);
}

void SpartaGui::writeRecoveryFile()
{
    // only for unsaved changes; never touches the user's real file
    if (!textEdit->document()->isModified()) return;
    const QString text = textEdit->toPlainText();
    if (text.trimmed().isEmpty()) return;

    const QString path = recoveryFilePath();
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        f.write(text.toUtf8());
        f.close();
    }
    // sidecar manifest recording where the buffer really belongs
    QFile meta(path + ".json");
    if (meta.open(QIODevice::WriteOnly)) {
        QJsonObject o;
        o["realPath"] = currentFile.isEmpty() ? QString()
                                              : QDir(currentDir).absoluteFilePath(currentFile);
        o["savedAt"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        meta.write(QJsonDocument(o).toJson());
    }
}

void SpartaGui::clearRecoveryFile()
{
    QFile::remove(recoveryFilePath());
    QFile::remove(recoveryFilePath() + ".json");
}

bool SpartaGui::maybeRecoverSession()
{
    const QString path = recoveryFilePath();
    if (!QFileInfo::exists(path)) return false;

    QString realPath, savedAt;
    QFile meta(path + ".json");
    if (meta.open(QIODevice::ReadOnly)) {
        const QJsonObject o = QJsonDocument::fromJson(meta.readAll()).object();
        realPath = o["realPath"].toString();
        savedAt = o["savedAt"].toString();
    }
    const QString what = realPath.isEmpty() ? "an unsaved buffer" : realPath;
    const auto btn = QMessageBox::question(
        this, "Recover Unsaved Work",
        QString("SPARTA-GUI found autosaved work from a previous session (%1%2).\n\n"
                "Recover it into the editor?")
            .arg(what, savedAt.isEmpty() ? "" : ", " + savedAt),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (btn != QMessageBox::Yes) {
        clearRecoveryFile();
        return false;
    }
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    textEdit->setPlainText(QString::fromUtf8(f.readAll()));
    if (!realPath.isEmpty()) {
        currentFile = QFileInfo(realPath).fileName();
        currentDir = QFileInfo(realPath).absolutePath();
        // and say so: with currentFile set, Save writes straight to that file,
        // so a title still reading *unknown* would have the user believe the
        // buffer is nameless right up to the point it overwrites their deck
        setWindowTitle(QString("SPARTA-GUI - Editor - " + currentFile));
    }
    textEdit->document()->setModified(true); // it is unsaved by definition
    showEditor();
    return true;
}

void SpartaGui::ensureHistory()
{
    if (history) return;
    history = new RunHistory(this);
    connect(history, &RunHistory::message, this,
            [this](const QString &m) { statusBar()->showMessage(m, 8000); });
}

void SpartaGui::ensureHistoryPanel()
{
    ensureHistory();
    if (historyPanel) return;
    historyPanel = new HistoryPanel(this, history);
    panels->setPanelWidget(PanelManager::History, historyPanel, "Run History");
}

void SpartaGui::showRunHistory()
{
    ensureHistoryPanel();
    panels->openPanel(PanelManager::History);
}

void SpartaGui::ensureDiagnosticsPanel()
{
    if (diagnosticsList) return;
    diagnosticsList = new QListWidget(this);
    diagnosticsList->setObjectName("diagnosticsList");
    diagnosticsList->setAlternatingRowColors(true);
    // jump to the flagged line when an entry is activated
    connect(diagnosticsList, &QListWidget::itemActivated, this, [this](QListWidgetItem *item) {
        if (!item) return;
        const int line = item->data(Qt::UserRole).toInt();
        if (line > 0) {
            textEdit->setCursor(line - 1);
            textEdit->setFocus();
        }
        // show the command's documented syntax, if any, in the status bar
        const QString syntax = item->data(Qt::UserRole + 1).toString();
        if (!syntax.isEmpty()) statusBar()->showMessage(QStringLiteral("Syntax: %1").arg(syntax));
    });
    panels->setPanelWidget(PanelManager::Diagnostics, diagnosticsList, "Diagnostics");
}

InputCheck::Context SpartaGui::buildCheckContext()
{
    InputCheck::Context ctx;

    // doc-derived per-command argument specs (bundled resource)
    QFile tf(QStringLiteral(":/command_syntax.table"));
    if (tf.open(QIODevice::ReadOnly | QIODevice::Text))
        ctx.commandSpecs = InputCheck::parseSyntaxTable(QString::fromUtf8(tf.readAll()));

    // merge the richer JSON catalog's keyword sets (+ where the keyword list
    // begins) into the specs so the validator can flag unknown keywords
    QFile jf(QStringLiteral(":/command_syntax.json"));
    if (jf.open(QIODevice::ReadOnly)) {
        const auto help = InputCheck::parseSyntaxCatalog(jf.readAll());
        for (auto it = help.constBegin(); it != help.constEnd(); ++it) {
            InputCheck::CommandSpec &spec = ctx.commandSpecs[it.key()];
            if (it.value().keywordStart >= 0 && !it.value().keywords.isEmpty()) {
                spec.keywordStart = it.value().keywordStart;
                spec.keywords = QSet<QString>(it.value().keywords.constBegin(),
                                              it.value().keywords.constEnd());
            }
            spec.numericArgs = it.value().numericArgs;
        }
    }

    for (auto it = ctx.commandSpecs.constBegin(); it != ctx.commandSpecs.constEnd(); ++it)
        ctx.commands.insert(it.key());

    // command names from the help index (2-token lines: "<page> <command>")
    QFile hf(QStringLiteral(":/help_index.table"));
    if (hf.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QStringList lines = QString::fromUtf8(hf.readAll()).split(QLatin1Char('\n'));
        for (const QString &line : lines) {
            const QStringList w = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
            if (w.size() == 2) ctx.commands.insert(w.at(1));
        }
    }

    // library-internal commands not covered by the help index
    QFile inf(QStringLiteral(":/sparta_internal_commands.txt"));
    if (inf.open(QIODevice::ReadOnly | QIODevice::Text)) {
        const QStringList lines = QString::fromUtf8(inf.readAll()).split(QLatin1Char('\n'));
        for (const QString &raw : lines) {
            const QString c = raw.trimmed();
            if (!c.isEmpty() && !c.startsWith(QLatin1Char('#'))) ctx.commands.insert(c);
        }
    }

    // Authoritative command + style names from the live SPARTA library, when an
    // instance exists.  (The help index only lists styles that have their own
    // doc page, so it is not a complete style dictionary -- style-name checks are
    // skipped entirely unless the library can supply the full list.)
    if (sparta.isOpen()) {
        auto libStyles = [this](const char *kind) {
            QSet<QString> out;
            const int n = sparta.styleCount(kind);
            for (int i = 0; i < n; ++i) {
                QString s = sparta.styleName(kind, i);
                if (s.isEmpty()) continue;
                if (s.endsWith("/kk/host") || s.endsWith("/kk/device") || s.endsWith("/kk"))
                    continue; // keep the base (non-accelerated) name only
                out.insert(s);
            }
            return out;
        };
        const QSet<QString> cmds = libStyles("command");
        ctx.commands.unite(cmds);
        static const char *const kinds[] = {"fix",     "compute",      "dump",
                                            "region",  "collide",      "react",
                                            "surf_collide", "surf_react"};
        for (const char *k : kinds) {
            const QSet<QString> s = libStyles(k);
            if (!s.isEmpty()) ctx.styles[QString::fromLatin1(k)] = s;
        }
    }

    // resolve deck-referenced files relative to the current working directory
    const QString dir = currentDir;
    ctx.fileExists = [dir](const QString &name) {
        if (name.isEmpty()) return true; // don't flag empty tokens
        QFileInfo fi(name);
        if (fi.isAbsolute()) return fi.exists();
        return QFileInfo(QDir(dir), name).exists();
    };
    return ctx;
}

void SpartaGui::checkInput()
{
    runInputCheck(true);
}

void SpartaGui::autoCheckInput()
{
    if (!autoLintEnabled) return;
    // only lint the editor, never the welcome screen; and skip while a run is
    // active (SPARTA is not re-entrant and the buffer is being executed anyway)
    if (centralStack->currentWidget() != textEdit) return;
    if (sparta.isRunning()) return;
    runInputCheck(false);
}

void SpartaGui::runInputCheck(bool interactive)
{
    ensureDiagnosticsPanel();
    const InputCheck::Context ctx = buildCheckContext();
    const QStringList lines = textEdit->toPlainText().split(QLatin1Char('\n'));
    QList<InputCheck::Diagnostic> diags = InputCheck::checkDeck(lines, ctx);

    // Automatic linting must not flag the line the user is still typing on: a
    // half-written command (unknown command, too few args, ...) would light up
    // mid-keystroke.  Suppress diagnostics on the current cursor line for the
    // auto pass -- they appear as soon as the cursor moves to another line.
    // The manual "Check Input" (interactive) always reports every line.
    if (!interactive) {
        const int curLine = textEdit->textCursor().blockNumber() + 1; // 1-based
        diags.erase(std::remove_if(diags.begin(), diags.end(),
                                   [curLine](const InputCheck::Diagnostic &d) {
                                       return d.line == curLine;
                                   }),
                    diags.end());
    }

    textEdit->setDiagnostics(diags);

    // fill the diagnostics panel
    diagnosticsList->clear();
    int errors = 0, warnings = 0;
    for (const auto &d : diags) {
        if (d.severity == InputCheck::Severity::Error) ++errors;
        else if (d.severity == InputCheck::Severity::Warning) ++warnings;
        const QString sev = (d.severity == InputCheck::Severity::Error) ? QStringLiteral("error")
                            : (d.severity == InputCheck::Severity::Warning)
                                ? QStringLiteral("warning")
                                : QStringLiteral("info");
        auto *item = new QListWidgetItem(
            QStringLiteral("line %1: %2: %3").arg(d.line).arg(sev, d.message));
        item->setData(Qt::UserRole, d.line);
        item->setIcon(QIcon(d.severity == InputCheck::Severity::Error
                                ? QStringLiteral(":/icons/dialog-no.svg")
                                : QStringLiteral(":/icons/warning.svg")));
        // attach the command's documented syntax as error help (tooltip + on click)
        if (d.line >= 1 && d.line <= lines.size()) {
            const QString cmd = lines.at(d.line - 1).trimmed().section(
                QRegularExpression(QStringLiteral("\\s+")), 0, 0);
            const auto hit = textEdit->commandHelp().constFind(cmd);
            if (hit != textEdit->commandHelp().constEnd() && !hit.value().syntax.isEmpty()) {
                item->setData(Qt::UserRole + 1, hit.value().syntax);
                item->setToolTip(QStringLiteral("Syntax: %1").arg(hit.value().syntax));
            }
        }
        diagnosticsList->addItem(item);
    }

    if (diags.isEmpty()) {
        diagnosticsList->addItem(new QListWidgetItem(QStringLiteral("No problems found.")));
        if (interactive) statusBar()->showMessage("Input check: no problems found.", 5000);
    } else {
        // the manual check raises the panel and summarizes; the auto-lint path
        // updates the inline markers + panel contents quietly (no focus stealing)
        if (interactive) panels->openPanel(PanelManager::Diagnostics);
        if (interactive)
            statusBar()->showMessage(
                QStringLiteral("Input check: %1 error(s), %2 warning(s).")
                    .arg(errors)
                    .arg(warnings),
                8000);
    }
}

void SpartaGui::ensureProjectFilesPanel()
{
    if (projectFilesList) return;
    projectFilesList = new QListWidget(this);
    projectFilesList->setObjectName("projectFilesList");
    projectFilesList->setToolTip("Files in the working directory; bold entries are referenced "
                                 "by the current deck. Double-click to open.");
    connect(projectFilesList, &QListWidget::itemActivated, this, [this](QListWidgetItem *item) {
        if (item && !item->data(Qt::UserRole).toString().isEmpty())
            openFile(item->data(Qt::UserRole).toString());
    });
    // Qt-ADS hides the tab of a dock that is alone in its area, and this is the
    // only panel with an area to itself: it came up as a title bar of buttons
    // with no name anywhere on it. Nothing in the dock's API turns that back
    // on, so the panel says what it is itself -- which the Parameter Sweep
    // panel does anyway, for its own reasons.
    auto *page   = new QWidget(this);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);
    auto *heading = new QLabel("Project Files");
    QFont headingFont = heading->font();
    headingFont.setBold(true);
    heading->setFont(headingFont);
    heading->setContentsMargins(4, 2, 4, 2);
    layout->addWidget(heading);
    layout->addWidget(projectFilesList);

    panels->setPanelWidget(PanelManager::ProjectFiles, page, "Project Files");
}

namespace {
// Icon for an entry of the project file list.  These are files, so the
// document-open glyph the list used for every one of them -- an opening folder,
// the icon of the *action* that opens a file -- was wrong for all of them and
// told the reader nothing about what any entry was.
QString projectFileIcon(const QFileInfo &fi)
{
    const QString name = fi.fileName();
    const QString ext  = fi.suffix().toLower();

    if (name.startsWith("in.")) return ":/icons/run-file.svg";
    if (name.startsWith("log.") || ext == "log" || ext == "txt")
        return ":/icons/txt-file-icon.svg";
    if (isMovieFile(name) || isImageFile(name)) return ":/icons/image-x-generic.svg";
    if (ext == "csv") return ":/icons/csv-file-icon.svg";
    if (ext == "yaml" || ext == "yml") return ":/icons/yaml-file-icon.svg";
    // restart files are named <base>.<step> or .restart, and are binary
    if (ext == "restart" || name.contains(".restart")) return ":/icons/binary-file-icon.svg";
    return ":/icons/txt-file-icon.svg";
}
} // namespace

void SpartaGui::refreshProjectFiles()
{
    if (!projectFilesList) return;
    projectFilesList->clear();

    // files referenced by the current deck via include / read_* commands
    static const QRegularExpression readRe(
        QStringLiteral("^\\s*(include|read_surf|read_grid|read_restart|read_particles|read_isurf)"
                       "\\s+(\\S+)"),
        QRegularExpression::MultilineOption);
    QSet<QString> referenced;
    auto it = readRe.globalMatch(textEdit->toPlainText());
    while (it.hasNext()) {
        const QString f = it.next().captured(2);
        if (!f.contains(QLatin1Char('$')) && !f.contains(QLatin1Char('*'))) referenced.insert(f);
    }

    const QDir dir(currentDir.isEmpty() ? QDir::currentPath() : currentDir);
    const QFileInfo curInfo(currentFile);
    const QString curName = curInfo.fileName();
    const QFileInfoList entries =
        dir.entryInfoList(QDir::Files, QDir::Name | QDir::IgnoreCase);
    for (const QFileInfo &fi : entries) {
        const QString name = fi.fileName();
        auto *item = new QListWidgetItem(name);
        item->setData(Qt::UserRole, fi.absoluteFilePath());
        item->setIcon(QIcon(projectFileIcon(fi)));
        // emphasize files the deck references, and the currently open file
        if (referenced.contains(name)) {
            QFont f = item->font();
            f.setBold(true);
            item->setFont(f);
            item->setToolTip("Referenced by the current deck");
        }
        if (!curName.isEmpty() && name == curName)
            item->setSelected(true);
        projectFilesList->addItem(item);
    }
    if (projectFilesList->count() == 0)
        projectFilesList->addItem(new QListWidgetItem("(no files in working directory)"));
}

void SpartaGui::archiveFinishedRun(bool success)
{
    if (!QSettings().value(Keys::ARCHIVE_RUNS, false).toBool()) return;
    ensureHistory();

    RunArchive::RunRecord rec;
    rec.id = QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss") + "-" +
             QString::number(runCounter);
    rec.timestamp = QDateTime::currentDateTime().toString(Qt::ISODate);
    rec.deckName = currentFile.isEmpty() ? "buffer" : currentFile;
    rec.deckText = textEdit->toPlainText();
    rec.workDir = currentDir;
    rec.status = success ? "ok" : "failed";
    if (logwindow) rec.logText = logwindow->toPlainText();
    rec.metadata.insert("Run number", QString::number(runCounter));
    rec.metadata.insert("SPARTA version", QString::number(sparta.version()));

    // --- rigorous DOE provenance: capture the build + host environment so an
    //     archived run can be traced back to exactly what produced it ---

    // build provenance read from the running SPARTA library
    const QString verStr = sparta.versionString();
    if (!verStr.isEmpty()) rec.metadata.insert("SPARTA version date", verStr);
    const QString gitCommit = sparta.gitCommit();
    if (!gitCommit.isEmpty()) rec.metadata.insert("SPARTA git commit", gitCommit);
    const QString gitBranch = sparta.gitBranch();
    if (!gitBranch.isEmpty()) rec.metadata.insert("SPARTA git branch", gitBranch);

    rec.metadata.insert("Parallelism",
                        sparta.configHasMpiSupport() ? "MPI" : "serial");

    if (sparta.configHasPackage("KOKKOS")) {
        QStringList apis;
        for (const char *api : {"serial", "openmp", "cuda", "hip"})
            if (sparta.configAccelerator("KOKKOS", "api", api)) apis << api;
        rec.metadata.insert("Accelerator",
                            "KOKKOS (" + (apis.isEmpty() ? QString("?") : apis.join('/')) + ")");
    } else {
        rec.metadata.insert("Accelerator", "none");
    }

    {
        QStringList pkgs;
        for (const char *p : {"KOKKOS", "FFT"})
            if (sparta.configHasPackage(p)) pkgs << p;
        rec.metadata.insert("Packages", pkgs.isEmpty() ? QString("(none)") : pkgs.join(", "));
    }

    {
        QStringList io;
        if (sparta.configHasPngSupport()) io << "PNG";
        if (sparta.configHasJpegSupport()) io << "JPEG";
        if (sparta.configHasFfmpegSupport()) io << "FFmpeg";
        if (sparta.configHasGzipSupport()) io << "gzip";
        rec.metadata.insert("I/O support", io.isEmpty() ? QString("(none)") : io.join(", "));
    }

    // host / OS / environment (GUI side)
    rec.metadata.insert("Host", QSysInfo::machineHostName());
    rec.metadata.insert("OS", QSysInfo::prettyProductName());
    rec.metadata.insert("Kernel", QSysInfo::kernelType() + " " + QSysInfo::kernelVersion());
    rec.metadata.insert("Architecture", QSysInfo::currentCpuArchitecture());
    const QByteArray modules = qgetenv("LOADEDMODULES");
    if (!modules.isEmpty())
        rec.metadata.insert("Environment modules",
                            QString::fromLocal8Bit(modules).replace(':', ' '));
    rec.metadata.insert("Command line", QCoreApplication::arguments().join(' '));
    if (!currentDir.isEmpty()) rec.metadata.insert("Working directory", currentDir);

    QStringList images;
    if (viewer && viewer->sequence()) images = viewer->sequence()->images();
    history->archive(rec, images);
}

void SpartaGui::exportParaview()
{
    // pre-fill the file pickers from the current deck's directory
    QString deckDir = currentDir;
    if (deckDir.isEmpty() && !currentFile.isEmpty())
        deckDir = QFileInfo(currentFile).absolutePath();
    if (deckDir.isEmpty()) deckDir = QDir::currentPath();

    ParaViewExportDialog dlg(this, deckDir);
    dlg.exec();
}

void SpartaGui::surfaceReport()
{
    if (sparta.extractSetting("surf_exist") != 1) {
        warning(this, "Surface Quantities Report",
                "No surfaces exist yet.  Run a deck that reads a surface and defines "
                "a per-surf compute (e.g. \"compute 1 surf all all fx fy fz etot\"), "
                "then open this report.");
        return;
    }
    SurfReportDialog dlg(this, &sparta, textEdit->toPlainText());
    dlg.exec();
}

void SpartaGui::continueRestart()
{
    QDir dir(currentDir.isEmpty() ? QDir::currentPath() : currentDir);
    // Deliberately narrow: a bare "*restart*" glob also matches log files and
    // notes that merely mention the word, offering them as restart candidates.
    const QStringList filters = {"*.restart", "*.restart.*", "*.spart"};
    const QFileInfoList found = dir.entryInfoList(filters, QDir::Files, QDir::Time);

    QDialog dlg(this);
    dlg.setWindowTitle("Insert Restart Commands");
    dlg.resize(680, 420);
    auto *outer = new QVBoxLayout(&dlg);
    outer->addWidget(new QLabel("Restart files in the working directory. The selected file\nbecomes a read_restart + run pair inserted into the editor for review:", &dlg));
    auto *list = new QListWidget(&dlg);
    auto addFile = [&](const QFileInfo &fi) {
        auto *it = new QListWidgetItem(QString("%1    (%2 KB, %3)")
                                           .arg(fi.fileName())
                                           .arg(fi.size() / 1024)
                                           .arg(fi.lastModified().toString("yyyy-MM-dd hh:mm")));
        it->setData(Qt::UserRole, fi.absoluteFilePath());
        list->addItem(it);
    };
    for (const auto &fi : found) addFile(fi);
    outer->addWidget(list, 1);

    auto *steprow = new QHBoxLayout;
    steprow->addWidget(new QLabel("Additional steps:", &dlg));
    auto *steps = new QSpinBox(&dlg);
    steps->setRange(0, 2000000000);
    steps->setValue(1000);
    steprow->addWidget(steps);
    steprow->addStretch();
    auto *browseBtn = new QPushButton("Browse...", &dlg);
    steprow->addWidget(browseBtn);
    outer->addLayout(steprow);

    auto *bb = new QDialogButtonBox(&dlg);
    auto *inspectBtn = bb->addButton("Inspect", QDialogButtonBox::ActionRole);
    bb->addButton("Insert Continue Commands", QDialogButtonBox::AcceptRole);
    bb->addButton(QDialogButtonBox::Close);
    outer->addWidget(bb);

    connect(browseBtn, &QPushButton::clicked, &dlg, [&]() {
        const QString f =
            QFileDialog::getOpenFileName(&dlg, "Select restart file", dir.absolutePath());
        if (f.isEmpty()) return;
        addFile(QFileInfo(f));
        list->setCurrentRow(list->count() - 1);
    });
    connect(inspectBtn, &QPushButton::clicked, &dlg, [&]() {
        if (auto *cur = list->currentItem()) inspectFile(cur->data(Qt::UserRole).toString());
    });
    connect(bb, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    if (list->count()) list->setCurrentRow(0);

    if (dlg.exec() != QDialog::Accepted) return;
    auto *cur = list->currentItem();
    if (!cur) {
        warning(this, "Insert Restart Commands", "No restart file selected.");
        return;
    }
    const QString path = cur->data(Qt::UserRole).toString();
    const QString rel = QDir(currentDir).relativeFilePath(path);
    // compose the minimal continue deck and insert it for review before running
    const QString cmds = QString("read_restart %1\nrun %2\n").arg(rel).arg(steps->value());
    showEditor();
    QTextCursor c = textEdit->textCursor();
    c.insertText(cmds);
    textEdit->setTextCursor(c);
    statusBar()->showMessage("Inserted read_restart + run; review and Run to continue.", 8000);
}

void SpartaGui::insertSnippet()
{
    const auto snips = Snippets::builtin();
    if (snips.isEmpty()) {
        warning(this, "Insert Snippet", "No snippets are available.");
        return;
    }

    QDialog dlg(this);
    dlg.setWindowTitle("Insert Snippet");
    dlg.resize(700, 460);
    auto *outer = new QVBoxLayout(&dlg);
    auto *split = new QSplitter(Qt::Horizontal, &dlg);
    auto *list = new QListWidget(split);
    for (int i = 0; i < snips.size(); ++i) {
        auto *it = new QListWidgetItem(QString("%1  —  %2").arg(snips[i].category, snips[i].name));
        it->setData(Qt::UserRole, i);
        it->setToolTip(snips[i].description);
        list->addItem(it);
    }
    auto *preview = new QPlainTextEdit(split);
    preview->setReadOnly(true);
    preview->setStyleSheet("QPlainTextEdit { font-family: monospace; }");
    split->addWidget(list);
    split->addWidget(preview);
    split->setStretchFactor(1, 2);
    outer->addWidget(split, 1);

    auto *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    bb->button(QDialogButtonBox::Ok)->setText("Insert at Cursor");
    outer->addWidget(bb);

    connect(list, &QListWidget::currentItemChanged, &dlg,
            [&, preview](QListWidgetItem *cur, QListWidgetItem *) {
                if (!cur) { preview->clear(); return; }
                const auto &s = snips[cur->data(Qt::UserRole).toInt()];
                preview->setPlainText(s.description + "\n\n" + s.body);
            });
    connect(bb, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    connect(list, &QListWidget::itemDoubleClicked, &dlg, &QDialog::accept);
    list->setCurrentRow(0);

    if (dlg.exec() != QDialog::Accepted) return;
    auto *cur = list->currentItem();
    if (!cur) return;
    const auto &s = snips[cur->data(Qt::UserRole).toInt()];
    showEditor();
    QTextCursor c = textEdit->textCursor();
    c.insertText(s.body + "\n");
    textEdit->setTextCursor(c);
}

void SpartaGui::importSurface()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Import Surface (STL or SPARTA surface file)", currentDir,
        // SPARTA's own examples name their surfaces data.circle, data.sphere and
        // so on, so a default filter of *.stl/*.surf shows an empty directory
        // for exactly the files most users are trying to import.
        "Surface geometry (*.stl *.surf data.*);;STL files (*.stl);;"
        "SPARTA surface files (*.surf data.*);;All files (*)");
    if (fileName.isEmpty()) return;

    // ensure a SPARTA instance exists so the wizard's "Render via SPARTA" preview
    // works (it needs the library to build the surf/grid and issue a dump image);
    // the STL parsing/transform/watertightness steps do not need it, but the
    // render tab otherwise reports "The SPARTA library is not loaded"
    if (!sparta.isRunning() && !sparta.isOpen()) startSparta();

    StlImportWizard wiz(this, &sparta, fileName);
    if (!wiz.loaded()) {
        critical(this, "Import Surface",
                 "Could not parse the selected file as an STL or SPARTA surface file.");
        return;
    }
    if (wiz.exec() != QDialog::Accepted) return;

    const QString text = wiz.generatedText();
    if (text.isEmpty()) return;
    showEditor();
    QTextCursor cursor = textEdit->textCursor();
    cursor.insertText(text + "\n");
    textEdit->setTextCursor(cursor);
}

void SpartaGui::plotDataFile()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open Data File to Plot", QString(),
        "Data files (*.dat *.csv *.yaml *.yml *.json *.txt);;All files (*)");
    if (fileName.isEmpty()) return;

    QString error;
    PlotData data = loadPlotData(fileName, &error);
    if (data.isEmpty()) {
        critical(this, "Plot Data File",
                 "Could not read data from file:", error.isEmpty() ? fileName : error);
        return;
    }

    PlotDataDialog dialog(data, this);
    if (dialog.exec() != QDialog::Accepted) return;
    const QList<int> ycols = dialog.yColumns();
    if (ycols.isEmpty()) {
        warning(this, "Plot Data File", "No data columns were selected to plot.");
        return;
    }

    const PlotData plotData = dialog.buildData();

    // standalone chart window (no live simulation); cleans itself up on close
    auto *win = new ChartWindow(fileName, nullptr);
    win->setAttribute(Qt::WA_DeleteOnClose);
    win->setWindowTitle(QString("Plot: %1 - SPARTA-GUI").arg(QFileInfo(fileName).fileName()));
    win->setWindowIcon(QIcon(Cfg::MAIN_ICON));
    win->setMinimumSize(Cfg::MINIMUM_WIDTH, Cfg::MINIMUM_HEIGHT);
    win->loadData(plotData, dialog.xColumn(), ycols);
    win->show();
}

void SpartaGui::renderImage(bool quiet)
{
    // "quiet" is for the renders nobody asked for: opening the viewer panel, or
    // entering a workspace that shows it, renders so the pane is not blank. A
    // deck that cannot be rendered is a perfectly ordinary state for that path
    // -- an empty buffer, or one that never creates a box -- and answering a
    // workspace switch with a modal error dialog is not acceptable. An explicit
    // Create Image still reports why nothing appeared.
    auto complain = [this, quiet](const QString &title, const QString &text,
                                  const QString &detail = QString()) {
        if (!quiet) warning(this, title, text, detail);
    };

    // SPARTA is not re-entrant, so we can only query SPARTA when it is not running
    if (!sparta.isRunning()) {
        startSparta();
        if (!sparta.extractSetting("box_exist")) {
            // there is no current system defined yet.
            // so we select the input from the start to the first run command
            // add a run 0 and thus create the state of the initial system without running.
            // this will allow us to create a snapshot image.
            auto saved = textEdit->textCursor();
            textEdit->moveCursor(QTextCursor::Start);
            if (textEdit->find(QRegularExpression(QStringLiteral(R"(^\s*run\s+)")))) {
                auto cursor = textEdit->textCursor();
                cursor.movePosition(QTextCursor::PreviousBlock);
                cursor.movePosition(QTextCursor::EndOfLine);
                cursor.movePosition(QTextCursor::Start, QTextCursor::KeepAnchor);
                auto selection = cursor.selectedText().replace(QChar(0x2029), '\n');
                selection += "\nrun 0 pre yes post no";
                textEdit->setTextCursor(saved);
                {
                    StdoutSilencer guard;
                    sparta.command("clear");
                    clearVariables();
                    sparta.commandsString(selection);
                }

                const QString errmsg = sparta.lastErrorMessage();
                // ignore "Invalid SPARTA handle", but report other errors
                if (!errmsg.isEmpty() && !errmsg.contains("Invalid SPARTA handle")) {
                    complain("Image Viewer File Creation Error",
                             "SPARTA failed to create the image:",
                             QString("<br><code>%1</code>").arg(errmsg));
                    return;
                }
            }
            textEdit->setTextCursor(saved);
            // still no system box. bail out with a suitable message
            if (!sparta.extractSetting("box_exist")) {
                complain("Image Viewer File Creation Error",
                         "Cannot create snapshot image from an input not creating a system box");
                return;
            }
        }

        // Purge the input deck's dump instances before opening the viewer: it
        // renders by creating its own dump and issuing "run 0", and leaving the
        // deck's dumps active would make that run re-trigger them and overwrite
        // their output files. (The walltime timeout the stop button leaves behind
        // is cleared per render in ImageViewer::createImage.)
        {
            StdoutSilencer guard;
            const int ndumps = sparta.idCount("dump");
            QStringList dumpids;
            for (int i = 0; i < ndumps; ++i)
                dumpids << sparta.idName("dump", i);
            for (const auto &id : dumpids)
                sparta.command("undump " + id);
        }

        ensureViewerPanel();
        // Keeping the previous render used to spawn an extra archived dock tab.
        // A sequence of renders is what the frame view is for, so an old one is
        // appended there instead and the panel keeps a single snapshot page.
        if (!QSettings().value(Keys::IMAGEREPLACE, true).toBool() && viewer->snapshot()) {
            const QImage kept = viewer->snapshot()->currentImage();
            const QString path =
                QDir::tempPath() + QString("/%1.kept.%2.png").arg(QFileInfo(currentFile).fileName()).arg(++keptRenderSeq);
            if (!kept.isNull() && kept.save(path)) {
                if (!viewer->sequence())
                    viewer->addSource(ViewerPanel::Sequence, new SlideShow(currentFile, this));
                viewer->sequence()->addImage(path);
            }
        }
        viewer->addSource(ViewerPanel::Snapshot, new ImageViewer(currentFile, &sparta, this));
    } else {
        complain("Image Viewer File Creation Error",
                 "Cannot create snapshot image while SPARTA is running");
        return;
    }
    panels->openPanel(PanelManager::Viewer);
    // an explicit Create Image is a request to look at the render
    viewer->showSource(ViewerPanel::Snapshot, true);
}

#if defined(SPARTA_GUI_HAVE_VTK)
void SpartaGui::open3DViewer()
{
    if (!sceneWindow) sceneWindow = new SceneWindow(this);
    sceneWindow->showViewer();
}

void SpartaGui::renderVtkSnapshot()
{
    if (sparta.isRunning()) {
        warning(this, "3D Snapshot", "Cannot create a 3D snapshot while SPARTA is running.");
        return;
    }
    if (!sceneWindow) sceneWindow = new SceneWindow(this);
    sceneWindow->showViewer();
    fillVtkScene(sceneWindow->scene(), /*quiet=*/false);
}

// Render the current state to VTK files and load them into @p scene.  Shared by
// the explicit "3D Snapshot" action, which reports what went wrong, and by the
// automatic refreshes of the docked 3D view, which must stay silent: a deck with
// no surfaces is not an error worth a dialog every time a run ends.
//
// Only the categories the scene's toggles ask for are rendered.  That is what
// ties the Particles/Grid/Surfaces buttons to the "dump ... /vtk" commands
// emitted below -- a category turned off is one SPARTA is never asked to write,
// rather than one written and then hidden.
int SpartaGui::fillVtkScene(VtkScene *scene, bool quiet)
{
    if (!scene || sparta.isRunning()) return 0;
    startSparta();

    // does the loaded library actually provide the VTK dump styles?  (VTK is an
    // optional SPARTA package; a stock build will not have them.)  If not, just
    // open the viewer so the user can still load dump-vtk files written elsewhere.
    bool haveVtkDump = false;
    const int ndumpstyles = sparta.styleCount("dump");
    for (int i = 0; i < ndumpstyles; ++i)
        if (sparta.styleName("dump", i).endsWith("/vtk")) { haveVtkDump = true; break; }
    if (!haveVtkDump) {
        if (!quiet)
            warning(this, "3D Snapshot",
                    "This SPARTA library was built without the VTK package,",
                    "so it cannot write VTK files directly.  You can still open <code>.vtu</code> / "
                    "<code>.vtp</code> files written by a VTK-enabled SPARTA build (or the "
                    "<i>Export to ParaView</i> tools) with the viewer's <b>Open</b> button.");
        return 0;
    }

    // ensure a system box exists, creating it with a "run 0" preflight of the
    // deck up to its first run command (same approach as renderImage()).
    if (!sparta.extractSetting("box_exist")) {
        auto saved = textEdit->textCursor();
        textEdit->moveCursor(QTextCursor::Start);
        if (textEdit->find(QRegularExpression(QStringLiteral(R"(^\s*run\s+)")))) {
            auto cursor = textEdit->textCursor();
            cursor.movePosition(QTextCursor::PreviousBlock);
            cursor.movePosition(QTextCursor::EndOfLine);
            cursor.movePosition(QTextCursor::Start, QTextCursor::KeepAnchor);
            auto selection = cursor.selectedText().replace(QChar(0x2029), '\n');
            selection += "\nrun 0 pre yes post no";
            textEdit->setTextCursor(saved);
            {
                StdoutSilencer guard;
                sparta.command("clear");
                clearVariables();
                sparta.commandsString(selection);
            }
        }
        textEdit->setTextCursor(saved);
        if (!sparta.extractSetting("box_exist")) {
            if (!quiet)
                warning(this, "3D Snapshot",
                        "Cannot create a 3D snapshot from an input that does not create a "
                        "system box.");
            return 0;
        }
    }

    // purge the deck's own dumps so our "run 0" does not re-trigger them
    {
        StdoutSilencer guard;
        const int ndumps = sparta.idCount("dump");
        QStringList dumpids;
        for (int i = 0; i < ndumps; ++i) dumpids << sparta.idName("dump", i);
        for (const auto &id : dumpids) sparta.command("undump " + id);
    }

    // Render one category (grid / particles / surfaces) to a temp VTK file, each
    // in its own isolated "run 0", so an empty or invalid category (e.g. surf/vtk
    // on a deck with no surfaces) cannot spoil the others.  Returns the produced
    // file path, or empty on failure.
    const QString dir = QDir::tempPath();
    auto renderCategory = [&](const QString &id, const QString &style, const QString &ext,
                              const QString &attrs) -> QString {
        const QString file = QString("%1/%2.0.%3").arg(dir, id, ext);
        QFile::remove(file);
        StdoutSilencer guard;
        if (sparta.hasId("dump", id.toLocal8Bit())) sparta.command("undump " + id);
        sparta.command(
            QString("dump %1 %2 1 %3.*.%4 %5").arg(id, style, QString("%1/%2").arg(dir, id), ext, attrs));
        const QString derr = sparta.lastErrorMessage();
        if (!derr.isEmpty() && !derr.contains("Invalid SPARTA handle")) {
            if (sparta.hasId("dump", id.toLocal8Bit())) sparta.command("undump " + id);
            return QString();
        }
        sparta.command("run 0 pre yes post no");
        (void)sparta.lastErrorMessage();
        if (sparta.hasId("dump", id.toLocal8Bit())) sparta.command("undump " + id);
        return QFileInfo::exists(file) ? file : QString();
    };

    struct Cat {
        QString id, style, ext, attrs, label;
        VtkScene::Kind kind;
    };
    const QList<Cat> cats = {
        {"sgvtkgrid", "grid/vtk", "vtu", "all id proc", "grid", VtkScene::Kind::Grid},
        {"sgvtkpart", "particle/vtk", "vtp", "all id x y z vx vy vz", "particles",
         VtkScene::Kind::Particles},
        {"sgvtksurf", "surf/vtk", "vtp", "all id type", "surfaces", VtkScene::Kind::Surface},
    };

    scene->clearScene();
    int loaded = 0;
    for (const auto &c : cats) {
        if (!scene->kindVisible(c.kind)) continue;
        const QString f = renderCategory(c.id, c.style, c.ext, c.attrs);
        if (f.isEmpty()) continue;
        if (scene->addDatasetFile(f, c.label, c.kind, nullptr)) ++loaded;
        QFile::remove(f);
    }

    if (loaded == 0 && !quiet)
        warning(this, "3D Snapshot",
                "No particle, grid or surface data was produced for the current state.");
    return loaded;
}

// Put whatever the deck currently describes into the docked 3D view.  Called
// when the Visualize workspace is entered and again when a run finishes, so the
// view is never the empty scene with one dead tab that it used to be: before a
// run that is the box and any surfaces the input reads, and afterwards it is
// the particles and grid as well.
void SpartaGui::refreshDocked3DScene()
{
    if (!viewer) return;
    auto *scene = qobject_cast<VtkScene *>(viewer->source(ViewerPanel::Scene));
    if (!scene) return;
    fillVtkScene(scene, /*quiet=*/true);
}
#endif // SPARTA_GUI_HAVE_VTK

void SpartaGui::clearPanelWidgets()
{
    // PanelManager::clearRunPanels() deletes each panel's inner widget, so every
    // pointer we hold to one dangles afterwards. Keeping that bookkeeping in a
    // single place means adding a panel cannot leave a stale pointer behind at
    // one of the call sites.
    panels->clearRunPanels();
    chartwindow      = nullptr;
    logwindow        = nullptr;
    viewer           = nullptr;
    varwindow        = nullptr;
    diagnosticsList  = nullptr;
    projectFilesList = nullptr;
    sweepPanel       = nullptr;
    historyPanel     = nullptr;

    // Put an empty output back straight away, and re-open it if the workspace
    // we are in is one that shows it. Neither caller re-applies the mode after
    // clearing, so without this, opening a file left the Setup workspace --
    // whose whole point is the deck beside its output -- as a bare editor
    // taking the entire window, and nothing brought the panel back until the
    // user ran something.
    ensureLogPanel();
    if (PanelManager::modeShows(panels->currentMode(), PanelManager::Log))
        panels->openPanel(PanelManager::Log);
}

void SpartaGui::createVariableWindow()
{
    varwindow = new QLabel(QString());
    varwindow->setText("(none)");

    varwindow->setFont(monoFontFromSettings());

    varwindow->setFrameStyle(QFrame::Sunken);
    varwindow->setFrameShape(QFrame::Panel);
    varwindow->setAlignment(Qt::AlignVCenter);
    varwindow->setContentsMargins(5, 5, 5, 5);
    varwindow->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    panels->setPanelWidget(PanelManager::Variables, varwindow, "Variables");
}

void SpartaGui::autoSave()
{
    // no need to auto-save, if the document has no name or is not modified.
    QString fileName = currentFile;
    if (fileName.isEmpty()) return;
    if (!textEdit->document()->isModified()) return;

    // check preference
    bool autosave = false;
    QSettings settings;
    settings.beginGroup(Keys::GROUP_REFORMAT);
    autosave = settings.value(Keys::AUTOSAVE, false).toBool();
    settings.endGroup();

    if (autosave) writeFile(fileName);
}

void SpartaGui::setFont(const QFont &newFont)
{
    QMainWindow::setFont(newFont);
    if (textEdit) {
        textEdit->setFont(newFont);
        menubar->setFont(newFont);
    }
}

void SpartaGui::about()
{
    std::string version = "<b>This is SPARTA-GUI version " SPARTA_GUI_VERSION;
    version += " using Qt version " QT_VERSION_STR;
    if (isLightTheme())
        version += " with light theme";
    else
        version += " with dark theme";
    version += "</b><br><br>\n";
    if (sparta.hasPlugin()) {
        version += "SPARTA library loaded as plugin";
        if (!pluginPath.isEmpty()) {
            version += " from file ";
            version += pluginPath.toStdString();
        }
    } else {
        version += "SPARTA library linked to executable";
    }

    QString to_clipboard(version.c_str());
    to_clipboard += "\n\n";

    QString info = "SPARTA is currently running. SPARTA config info not available.\n";
    QString details;

    // SPARTA is not re-entrant, so we can only query SPARTA when it is not running.
    // SPARTA has no "info" command, so the version and the available styles are
    // queried through the SPARTA library interface instead.
    if (!sparta.isRunning()) {
        startSparta();
        info.clear();
        const auto *verstr = static_cast<const char *>(sparta.extractGlobal("sparta_version"));
        if (verstr) info += QString("SPARTA version: %1\n").arg(verstr);
        info += QString("KOKKOS package: %1\n")
                    .arg(sparta.configHasPackage("KOKKOS") ? "included" : "not included");
        info += QString("PNG image support: %1\n")
                    .arg(sparta.configHasPngSupport() ? "yes" : "no");
        info += QString("JPEG image support: %1\n")
                    .arg(sparta.configHasJpegSupport() ? "yes" : "no");

        // build a listing of the available styles for each category
        auto styleInfo = [&](const char *category, const char *label) {
            QStringList styles;
            const int nstyles = sparta.styleCount(category);
            for (int i = 0; i < nstyles; ++i) {
                const QString style = sparta.styleName(category, i);
                if (!style.isEmpty()) styles << style;
            }
            styles.sort();
            return QString("%1 styles:\n%2\n\n").arg(label, styles.join(' '));
        };
        details += styleInfo("collide", "Collide");
        details += styleInfo("react", "React");
        details += styleInfo("surf_collide", "Surface collide");
        details += styleInfo("surf_react", "Surface react");
        details += styleInfo("compute", "Compute");
        details += styleInfo("fix", "Fix");
        details += styleInfo("dump", "Dump");
        details += styleInfo("region", "Region");
        details += styleInfo("command", "Command");
    }

    info += citeme;
    to_clipboard += info;
    to_clipboard += details;

#if QT_CONFIG(clipboard)
    if (auto *clip = QGuiApplication::clipboard()) clip->setText(to_clipboard);
#endif

    auto fsize = QFontMetrics(QApplication::font())
                     .size(Qt::TextSingleLine, "SPARTA-GUI configuration information line width");
    AboutDialog dialog(QString::fromStdString(version).trimmed(), info.trimmed(),
                       details.trimmed(), fsize.width(), this);
    dialog.exec();
}

#if defined(SPARTA_GUI_USE_PLUGIN)
void SpartaGui::checkUpdate()
{
    const auto libPath = LibraryAcquire::downloadDestination();
    const auto dlUrl   = getSpartaDownloadUrl();

    if (dlUrl.isEmpty()) {
        information(this, "Check for SPARTA Update",
                    "The pre-compiled SPARTA shared libraries from the SPARTA webserver "
                    "are not compatible with this SPARTA-GUI executable. Please compile "
                    "a matching SPARTA shared library yourself and select it in the "
                    "preferences dialog.");
        return;
    }

    if (!QFile::exists(libPath)) {
        information(this, "Check for SPARTA Update",
                    "No pre-compiled SPARTA library found in the configuration folder. "
                    "Click on 'Download SPARTA shared library' in the preferences dialog "
                    "to download one.");
        return;
    }

    URLDownloader downloader(this);
    QString expectedHash = downloader.getRemoteChecksum(dlUrl);
    if (expectedHash.isEmpty()) {
        critical(this, "Check for SPARTA Update", "Failed to retrieve remote checksum.",
                 downloader.errorString());
        return;
    }

    QString actualHash = URLDownloader::getLocalChecksum(libPath);
    if (actualHash == expectedHash) {
        information(this, "Check for SPARTA Update",
                    "Your downloaded SPARTA shared library is up-to-date.");
        return;
    } else {
        QMessageBox mb(this);
        mb.setWindowTitle("Check for SPARTA Shared Library Update");
        mb.setText("An updated pre-compiled SPARTA shared library is available. ");
        mb.setInformativeText("Do you want to download it now?");
        mb.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        mb.setWindowIcon(QIcon(Cfg::MAIN_ICON));
        mb.setIconPixmap(QPixmap(":/icons/sparta-plugin.png").scaled(96, 96));

        // customize button icons
        auto *button = mb.button(QMessageBox::Yes);
        button->setIcon(QIcon(":/icons/dialog-ok.svg"));
        button = mb.button(QMessageBox::No);
        button->setIcon(QIcon(":/icons/dialog-no.svg"));

        if (mb.exec() == QMessageBox::Yes) {
            QString reason;
            const auto got = LibraryAcquire::download(this, libPath, &reason);
            if (got == LibraryAcquire::Result::Ok) {
                warning(this, "SPARTA Shared Library Updated",
                        "The latest SPARTA library has been downloaded successfully. "
                        "SPARTA-GUI must be relaunched to activate it.");
                relaunchApplication();
            } else if (got == LibraryAcquire::Result::Failed) {
                critical(this, "Check for SPARTA Update",
                         "Failed to download SPARTA shared library.", reason);
            }
        }
        return;
    }
}
#endif

void SpartaGui::help()
{
    // the old Quick Help was one unscrollable QMessageBox of six dense
    // paragraphs; the same facts now live in task-shaped sections on the
    // first page of the help dialog, beside the generated shortcut list
    if (!helpsheet) helpsheet = new ShortcutsDialog(menubar, this);
    helpsheet->popup(ShortcutsDialog::GettingStarted);
}

void SpartaGui::showShortcuts()
{
    if (!helpsheet) helpsheet = new ShortcutsDialog(menubar, this);
    helpsheet->popup(ShortcutsDialog::Shortcuts);
}

void SpartaGui::showPalette()
{
    if (!palette) palette = new CommandPalette(menubar, this);
    palette->popup();
}

void SpartaGui::toggleAutoLint()
{
    autoLintEnabled = autoLintAction && autoLintAction->isChecked();
    QSettings().setValue(Keys::AUTOLINT, autoLintEnabled);
    // switching it on mid-edit should check what is already there
    if (autoLintEnabled) autoCheckInput();
}

void SpartaGui::manual()
{
    // the SPARTA online manual is not versioned
    QDesktopServices::openUrl(QUrl(Cfg::DOCS_URL + "/doc/Manual.html"));
}

void SpartaGui::howto()
{
    QDesktopServices::openUrl(QUrl("https://sparta.github.io/sparta-gui/"));
}

void SpartaGui::defaults()
{
    // Everything the user has configured goes, irreversibly -- which deserves a
    // question, not least because this entry sits one line under "Preferences".
    if (QMessageBox::question(
            this, "SPARTA-GUI - Reset Preferences",
            "Reset all preferences to their defaults?\n\n"
            "Fonts, paths, layout and editor settings will be lost. The location "
            "of the SPARTA shared library is kept.",
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes)
        return;

    QSettings settings;
    // Keep the library location: it is machine configuration, not a preference,
    // and wiping it used to drop the next launch into the missing-library
    // dialog -- a trap for anyone who only wanted their fonts back.
    const QString plugin = settings.value(Keys::PLUGIN_PATH).toString();
    settings.clear();
    if (!plugin.isEmpty()) settings.setValue(Keys::PLUGIN_PATH, plugin);
    settings.sync();
}

void SpartaGui::editVariables()
{
    // Re-read the buffer first: the list was last built when the file was
    // opened, so index variables added, removed or re-valued since then were
    // not in the dialog at all.  Values the user has already set are kept.
    updateVariables(true);

    QList<QPair<QString, QString>> newvars = variables;
    SetVariables vars(newvars, scriptVariables, this);
    vars.setFont(font());
    if (vars.exec() == QDialog::Accepted) {
        variables = newvars;
        stopAndReapRunner();
        {
            StdoutSilencer guard;
            sparta.close();
        }
        spartastatus->hide();
    }
}

void SpartaGui::findAndReplace()
{
    FindAndReplace find(textEdit, this);
    find.setFont(font());
    find.setObjectName("find");
    find.exec();
}

void SpartaGui::applyEditorColorScheme()
{
    const QString scheme =
        QSettings().value(Keys::COLOR_SCHEME, Highlighter::defaultScheme()).toString();
    const bool light = isLightTheme();
    if (highlighter) highlighter->applyScheme(scheme);
    if (textEdit)
        textEdit->setColorScheme(Highlighter::schemeBackground(scheme, light),
                                 Highlighter::schemeForeground(scheme, light));
}

void SpartaGui::preferences()
{
    // default settings are committed to QSettings during initialization of SPARTA-GUI
    QSettings settings;
    int oldthreads = settings.value(Keys::NTHREADS, 1).toInt();
    int oldaccel   = settings.value(Keys::ACCELERATOR, AcceleratorTab::None).toInt();
    bool oldecho   = settings.value(Keys::ECHO, false).toBool();

    Preferences prefs(&sparta, this);
    prefs.setFont(font());
    prefs.setObjectName("preferences");
    if (prefs.exec() == QDialog::Accepted) {
        // must delete SPARTA instance after preferences have changed that require
        // using different command line flags when creating the SPARTA instance like
        // suffixes or package commands
        int newthreads = settings.value(Keys::NTHREADS, nthreads).toInt();
        int newaccel   = settings.value(Keys::ACCELERATOR, AcceleratorTab::None).toInt();
        bool instanceClosed = false;
        if ((oldaccel != newaccel) || (oldthreads != newthreads) ||
            (oldecho != settings.value(Keys::ECHO, false).toBool())) {
            stopAndReapRunner();
            {
                StdoutSilencer guard;
                sparta.close();
            }
            instanceClosed = true;
            spartastatus->hide();
            // reset nthreads if accelerator does not support threads
            if (newaccel == AcceleratorTab::None)
                nthreads = 1;
            else
                nthreads = newthreads;

            qputenv("OMP_NUM_THREADS", QByteArray::number(nthreads));

            // Kokkos can be initialized only once per process, so once a run has
            // used it the thread count is fixed until SPARTA-GUI is restarted.
            // Tell the user rather than let a thread-count change silently do
            // nothing.
            if (kokkosStarted && (oldthreads != newthreads))
                warning(this, "Accelerator Settings",
                        "The number of threads cannot be changed after SPARTA has "
                        "run with the Kokkos accelerator, because Kokkos can be "
                        "initialized only once per process.",
                        "Restart SPARTA-GUI for the new thread count to take effect.");
        }
        // only refresh the snapshot if the instance is still alive: closing it
        // above tears down the box/grid, so a re-render would just pop a
        // "no simulation box" warning right after changing a preference.
        if (viewer && viewer->snapshot() && !instanceClosed)
            viewer->snapshot()->createImage();
        settings.beginGroup(Keys::GROUP_REFORMAT);
        textEdit->setReformatOnReturn(settings.value(Keys::RETURN, false).toBool());
        textEdit->setAutoComplete(settings.value(Keys::AUTOMATIC, true).toBool());
        const bool wasAutoLint = autoLintEnabled;
        autoLintEnabled        = settings.value(Keys::AUTOLINT, true).toBool();
        settings.endGroup();
        // if auto-lint was just turned off, drop any inline markers it left behind
        if (wasAutoLint && !autoLintEnabled) textEdit->clearDiagnostics();
        // the editor syntax color scheme may have changed: apply it live so the
        // choice takes effect immediately without requiring a restart
        applyEditorColorScheme();
        // the examples folder setting may have changed
        buildExampleMenu();
    }
}

void SpartaGui::appendAcceleratorArgs(int accel)
{
    // SPARTA only supports the KOKKOS accelerator package: -k on [t <N>] -sf kk
    if ((accel == AcceleratorTab::Kokkos) && sparta.configHasPackage("KOKKOS")) {
        spartaArgs.push_back("-k");
        spartaArgs.push_back("on");
        if (nthreads > 1) {
            spartaArgs.push_back("t");
            spartaArgs.push_back(std::to_string(nthreads));
        }
        spartaArgs.push_back("-sf");
        spartaArgs.push_back("kk");
    }
}

void SpartaGui::startSparta()
{
    // temporarily extend spartaArgs with additional arguments
    int initial_narg = spartaArgs.size();
    QSettings settings;
    int accel = settings.value(Keys::ACCELERATOR, AcceleratorTab::None).toInt();
    // if non-threaded accelerator selected reset threads
    if (accel == AcceleratorTab::None) {
        nthreads = 1;
    }
    qputenv("OMP_NUM_THREADS", QByteArray::number(nthreads));

    appendAcceleratorArgs(accel);

    if (settings.value(Keys::ECHO, false).toBool()) {
        spartaArgs.push_back("-echo");
        spartaArgs.push_back("screen");
    }

    // Build temporary char* array for the SPARTA C API which takes char**
    // but does not modify the argument strings. The const_cast is safe here
    // because sparta.open() only reads the strings to copy them internally.
    std::vector<char *> cargs;
    cargs.reserve(spartaArgs.size());
    for (auto &s : spartaArgs)
        cargs.push_back(const_cast<char *>(s.c_str()));
    int narg = static_cast<int>(cargs.size());
    sparta.open(narg, cargs.data());
    spartastatus->show();
    // Kokkos can be initialized at most once per process: record that it is now
    // live so a later thread-count change can tell the user a restart is needed.
    if (accel == AcceleratorTab::Kokkos) kokkosStarted = true;

    // A failed open is not a version problem, and saying so was actively
    // misleading.  version() answers 0 when there is no instance, so the test
    // below was trivially true whenever open() had failed: the user was told
    // their SPARTA was too old and the process exited, while the real reason sat
    // in lastErrorMessage() a few lines further down, which the exit made
    // unreachable.  The ordinary way to get here is the accelerator settings
    // adding "-k on ... -sf kk" arguments that the SPARTA constructor rejects.
    // startSparta() is also reached from the About dialog and the image render,
    // so opening About could end the session outright.
    if (!sparta.isOpen()) {
        const QString why = sparta.lastErrorMessage();
        spartaArgs.resize(initial_narg);
        critical(this, "SPARTA-GUI Error", "Error launching SPARTA:",
                 why.isEmpty() ? QStringLiteral("SPARTA could not be started with the "
                                                "current settings.")
                               : why);
        return;
    }

    if (sparta.version() < Cfg::MIN_SPARTA_VERSION) {
        critical(this, "SPARTA-GUI Error", "Incompatible SPARTA Version:",
                 "SPARTA-GUI version " SPARTA_GUI_VERSION " requires\n"
                 "a SPARTA version of at least " +
                     Cfg::MIN_SPARTA_VERSION_STR);
        exit(1);
    }

    // remove additional arguments (3 were there initially)
    spartaArgs.resize(initial_narg);

    const QString errmsg = sparta.lastErrorMessage();
    if (!errmsg.isEmpty()) critical(this, "SPARTA-GUI Error", "Error launching SPARTA:", errmsg);
}

bool SpartaGui::eventFilter(QObject *watched, QEvent *event)
{
    if (event->type() == QEvent::Close) {
        quit(); // quit() runs autoSave() itself
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

// Local Variables:
// c-basic-offset: 4
// End:
