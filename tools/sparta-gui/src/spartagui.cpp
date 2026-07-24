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
#include "chartviewer.h"
#include "codeeditor.h"
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
#include "runhistory.h"
#include "snippets.h"
#include "stlimportwizard.h"
#include "surfreportdialog.h"
#include "sweeppanel.h"
#include "setvariables.h"
#include "slideshow.h"
#include "stdcapture.h"
#include "urldownloader.h"
#if defined(SPARTA_GUI_HAVE_VTK)
#include "vtkviewer.h"
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
#include <QUrl>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

    // docked panel layout (Output/Charts/Image/Slide Show/Variables) replaces
    // setCentralWidget(): PanelManager installs the central stack as the
    // (non-closable) central dock itself
    panels = new PanelManager(this, centralStack);

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
        switch (mode) {
            case PanelManager::Setup:
                ensureDiagnosticsPanel();
                ensureProjectFilesPanel();
                refreshProjectFiles();
                panels->openPanel(PanelManager::ProjectFiles);
                panels->openPanel(PanelManager::Diagnostics);
                break;
            case PanelManager::RunMode:
                if (!varwindow) createVariableWindow();
                panels->openPanel(PanelManager::Variables);
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
    addMenuAction(menu, ":/icons/document-new.svg", "&New Input File", "Ctrl+N",
                  &SpartaGui::newDocument);
    addMenuAction(menu, ":/icons/document-open.svg", "&Open Input File", "Ctrl+O",
                  &SpartaGui::open);
    exampleMenu = menu->addMenu(QIcon(":/icons/document-open.svg"), "Open &Example");
    exampleMenu->setEnabled(false);
    addMenuAction(menu, ":/icons/document-save.svg", "&Save Input File", "Ctrl+S",
                  &SpartaGui::save);
    addMenuAction(menu, ":/icons/document-save-as.svg", "Save Input File &As", "Ctrl+Shift+S",
                  &SpartaGui::saveAs);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/txt-file-icon.svg", "&View Text File", "Ctrl+Shift+F",
                  &SpartaGui::view);
    addMenuAction(menu, ":/icons/image-x-generic.svg", "View &Image or Movie File(s)...",
                  "Ctrl+Shift+J", &SpartaGui::openImages);
    addMenuAction(menu, ":/icons/x-office-drawing.svg", "&Plot Data File...", "Ctrl+Shift+P",
                  &SpartaGui::plotDataFile);
    addMenuAction(menu, ":/icons/binary-file-icon.svg", "Inspect &Restart File", "Ctrl+Shift+R",
                  &SpartaGui::inspect);
    menu->addSeparator();

    recentActions.resize(Cfg::NUM_RECENT_FILES);
    for (int i = 0; i < Cfg::NUM_RECENT_FILES; ++i) {
        recentActions[i] = addMenuAction(menu, ":/icons/document-open-recent.svg",
                                         QString("&%1.").arg(i + 1), "", &SpartaGui::openRecent);
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

    addMenuAction(menu, ":/icons/preferences-desktop.svg", "P&references...", "Ctrl+P",
                  &SpartaGui::preferences);
    addMenuAction(menu, ":/icons/preferences-reset.svg", "Reset Preferences to &Defaults", "",
                  &SpartaGui::defaults);
}

void SpartaGui::createRunMenu()
{
    auto *menu = menubar->addMenu("&Run");
    addMenuAction(menu, ":/icons/system-run.svg", "&Run SPARTA from Editor Buffer", "Ctrl+Return",
                  &SpartaGui::runBuffer);
    addMenuAction(menu, ":/icons/run-file.svg", "Run SPARTA from &File", "Ctrl+Shift+Return",
                  &SpartaGui::runFile);
    addMenuAction(menu, ":/icons/process-stop.svg", "&Stop SPARTA", "Ctrl+/", &SpartaGui::stopRun);
    addMenuAction(menu, ":/icons/warning.svg", "Chec&k Input", "Ctrl+K", &SpartaGui::checkInput);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/system-restart.svg", "Relaunch &SPARTA Instance", "",
                  &SpartaGui::restartSparta);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/preferences-desktop.svg", "Set &Variables...", "Ctrl+Shift+V",
                  &SpartaGui::editVariables);
    addMenuAction(menu, ":/icons/binary-file-icon.svg", "Insert &Restart Commands...", "",
                  &SpartaGui::continueRestart);
    menu->addSeparator();

    addMenuAction(menu, ":/icons/image-viewer.svg", "Create &Image", "Ctrl+I",
                  &SpartaGui::renderImage);
#if defined(SPARTA_GUI_HAVE_VTK)
    addMenuAction(menu, ":/icons/image-viewer.svg", "3D &Snapshot (VTK)", "Ctrl+Shift+3",
                  &SpartaGui::renderVtkSnapshot);
#endif
}

void SpartaGui::createToolsMenu()
{
    // Geometry conversion, external export and reporting: work on simulation
    // data, but outside the edit-run-look loop that File and Run cover.
    auto *menu = menubar->addMenu("&Tools");
    addMenuAction(menu, ":/icons/vdw-style.svg", "Import Sur&face (STL / SPARTA)...", "Ctrl+Shift+T",
                  &SpartaGui::importSurface);
    addMenuAction(menu, ":/icons/image-x-generic.svg", "Export to Para&View...", "Ctrl+Shift+E",
                  &SpartaGui::exportParaview);
    addMenuAction(menu, ":/icons/vdw-style.svg", "Surface &Quantities Report...", "",
                  &SpartaGui::surfaceReport);
    menu->addSeparator();

    // Multi-run studies: each drives the same deck repeatedly, so they belong
    // together rather than mixed in with the single-run controls under Run.
    auto *studies = menu->addMenu(QIcon(":/icons/x-office-drawing.svg"), "&Studies");
    addMenuAction(studies, ":/icons/x-office-drawing.svg", "Parametric S&weep...", "",
                  &SpartaGui::runSweep);
    addMenuAction(studies, ":/icons/document-open-recent.svg", "Run &History...", "",
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
        {PanelManager::Setup, ":/icons/document-open.svg", "&Setup Workspace", "Ctrl+1"},
        {PanelManager::RunMode, ":/icons/system-run.svg", "&Run Workspace", "Ctrl+2"},
        {PanelManager::Analyze, ":/icons/x-office-drawing.svg", "&Analyze Workspace", "Ctrl+3"},
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
        {PanelManager::Image, ":/icons/image-viewer.svg", "&Image Window", "Ctrl+Shift+I"},
        {PanelManager::Slide, ":/icons/image-x-generic.svg", "&Slide Show Window", "Ctrl+L"},
        {PanelManager::Variables, ":/icons/utilities-terminal.svg", "&Variables Window",
         "Ctrl+Shift+W"},
        {PanelManager::Sweep, ":/icons/x-office-drawing.svg", "Parametric S&weep Window", ""},
        {PanelManager::History, ":/icons/document-open-recent.svg", "Run &History Window", ""},
        {PanelManager::Diagnostics, ":/icons/warning.svg", "&Diagnostics Window", ""},
        {PanelManager::ProjectFiles, ":/icons/document-open.svg", "Project &Files Window", ""},
    };
    for (const auto &e : entries) {
        auto *action = panels->toggleViewAction(e.panel);
        action->setIcon(QIcon(e.icon));
        action->setText(e.text);
        action->setShortcut(QKeySequence(e.shortcut));
        menu->addAction(action);
    }

    // persist only on user-driven toggles (QAction::triggered); run-driven
    // open/close of any panel (including the run-start slideshow hide) must
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
        if (panel == PanelManager::Slide && !slideshow) {
            slideshow = new SlideShow(currentFile, this);
            panels->setPanelWidget(PanelManager::Slide, slideshow, "Slide Show");
        }
        // opening the Image window with no snapshot yet used to show an empty
        // pane; render one on demand (renderImage() creates the viewer and
        // re-opens the panel -- the !imagewindow guard prevents recursion, and
        // if rendering is not possible it reports why instead of doing nothing).
        // Only on a genuine user open, though: during startup this same signal
        // fires from restoreLayout() -- before the SPARTA plugin is even loaded
        // -- and rendering then would call into an unloaded library and crash.
        if (panel == PanelManager::Image && !imagewindow && startupComplete) renderImage();
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
#if defined(SPARTA_GUI_HAVE_VTK)
    addMenuAction(menu, ":/icons/image-viewer.svg", "&3D Viewer (VTK)", "",
                  &SpartaGui::open3DViewer);
#endif
    addMenuAction(menu, ":/icons/help-faq.svg", "&Welcome Screen", "",
                  [this]() { showWelcome(); });
    addMenuAction(menu, ":/icons/preferences-reset.svg", "Reset &Layout", "",
                  [this]() { panels->resetCurrentMode(); });
}

void SpartaGui::createAboutMenu()
{
    auto *menu = menubar->addMenu("&About");
    addMenuAction(menu, ":/icons/sparta-gui-icon-128x128.png", "&About SPARTA-GUI", "Ctrl+Shift+A",
                  &SpartaGui::about);
    addMenuAction(menu, ":/icons/help-faq.svg", "Quick &Help", "Ctrl+Shift+H", &SpartaGui::help);
    addMenuAction(menu, ":/icons/system-help.svg", "SPARTA-&GUI Documentation", "Ctrl+Shift+G",
                  &SpartaGui::howto);
    addMenuAction(menu, ":/icons/help-browser.svg", "SPARTA Online &Manual", "Ctrl+Shift+M",
                  &SpartaGui::manual);

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

    // Workspace mode switch: a segmented control of checkable buttons. This is
    // the primary way to change what the window shows, so it lives permanently
    // on screen rather than only in the View menu.
    struct ModeBtn {
        PanelManager::Mode mode;
        const char *text;
        const char *tip;
    };
    static const ModeBtn modebtns[] = {
        {PanelManager::Setup, "Setup", "Prepare the input deck: project files and linter findings"},
        {PanelManager::RunMode, "Run", "Watch a run: console output, variables and live charts"},
        {PanelManager::Analyze, "Analyze", "Study results: charts, snapshot images and slide show"},
    };
    auto *modebar = new QWidget(this);
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
    statusbar->addWidget(modebar);

    auto *savebtn = new QPushButton(QIcon(":/icons/document-save.svg"), "");
    savebtn->setToolTip("Save edit buffer to file");
    connect(savebtn, &QPushButton::released, this, &SpartaGui::save);
    statusbar->addWidget(savebtn);

    auto *runbtn = new QPushButton(QIcon(":/icons/system-run.svg"), "");
    runbtn->setToolTip("Run SPARTA on input");
    connect(runbtn, &QPushButton::released, this, &SpartaGui::runBuffer);
    statusbar->addWidget(runbtn);

    auto *stopbtn = new QPushButton(QIcon(":/icons/process-stop.svg"), "");
    stopbtn->setToolTip("Stop SPARTA");
    connect(stopbtn, &QPushButton::released, this, &SpartaGui::stopRun);
    statusbar->addWidget(stopbtn);

    auto *imgbtn = new QPushButton(QIcon(":/icons/image-viewer.svg"), "");
    imgbtn->setToolTip("Create snapshot image");
    connect(imgbtn, &QPushButton::released, this, &SpartaGui::renderImage);
    statusbar->addWidget(imgbtn);

    // square status-bar buttons with a snug, uniform icon (shared policy)
    styleToolButtons(toolButtonSize(savebtn), {savebtn, runbtn, stopbtn, imgbtn});

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

    // set platform specific paths, library file name, config directory, and filename patterns
    QStringList dirlist{"."};
    const auto libName = getSpartaLibName();
#if defined(Q_OS_MACOS)
    const QString pattern = QStringLiteral("SPARTA shared library (libsparta*.dylib)");
    QStringList filter("libsparta*.dylib");
    dirlist.append(
        QString::fromLocal8Bit(qgetenv("DYLD_LIBRARY_PATH")).split(":", Qt::SkipEmptyParts));
    // library may be included in an application bundle:
    dirlist.append(QCoreApplication::applicationDirPath() + "/../Frameworks");
    dirlist.append({"/Applications/SPARTA-GUI.app/Contents/Frameworks",
                    "/Applications/SPARTA.app/Contents/Frameworks"});
#elif defined(Q_OS_WIN32)
    const QString pattern = QStringLiteral("SPARTA shared library (libsparta*.dll)");
    QStringList filter("libsparta*.dll");
    dirlist.append(QString::fromLocal8Bit(qgetenv("PATH")).split(";", Qt::SkipEmptyParts));
#else
    // for Linux and other unix-like systems
    const QString pattern = QStringLiteral("SPARTA shared library (libsparta*.so*)");
    QStringList filter("libsparta*.so*");
    dirlist.append(
        QString::fromLocal8Bit(qgetenv("LD_LIBRARY_PATH")).split(":", Qt::SkipEmptyParts));
#endif

    if (pluginPath.isEmpty()) {
        // construct list of possible standard choices for the shared library file
        // we prefer the current directory, then the dynamic library path, then some system folders
        // adapt file pattern and paths to the different operating systems

        // also check in the config dir location for a previously downloaded library
        dirlist.append(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation));
        // check some more system paths (only relevant for Linux and Unix-like
        // systems; they simply do not exist elsewhere)
        dirlist.append({"/usr/lib", "/usr/lib64", "/lib/x86_64-linux-gnu", "/usr/local/lib",
                        "/usr/local/lib64"});

        // construct list of matching files
        QFileInfoList entries;
        for (const auto &dir : dirlist)
            entries.append(QDir(dir).entryInfoList(filter));

        // convert list of paths to list of canonical file names
        QStringList choices;
        for (const auto &fn : entries)
            choices.append(fn.canonicalFilePath());
        choices.removeDuplicates();
        for (const auto &libpath : choices) {
            if (sparta.loadLib(libpath)) {
                pluginPath = libpath;
                settings.setValue(Keys::PLUGIN_PATH, pluginPath);
                settings.sync();
                break;
            }
        }

        // No suitable plugin was found automatically.  Show a dialog with three choices:
        // 1) Download a pre-compiled shared library from the SPARTA webserver
        //    (not offered when no compatible pre-compiled library exists, i.e. with MSVC)
        // 2) Browse the filesystem for a suitable shared library file
        // 3) Exit SPARTA-GUI
        const bool candownload = !getSpartaDownloadUrl().isEmpty();
        while (pluginPath.isEmpty()) {
            // remove key for path to the plugin so we won't get stuck in a loop reading a bad file
            settings.remove(Keys::PLUGIN_PATH);

            QMessageBox mb(this);
            mb.setWindowTitle("SPARTA-GUI - No SPARTA Shared Library");
            mb.setWindowIcon(QIcon(Cfg::MAIN_ICON));
            mb.setIconPixmap(QPixmap(":/icons/sparta-plugin.png").scaled(96, 96));
            mb.setText("No suitable SPARTA shared library found.");
            QString infotext =
                "<p align=\"justify\">Either the shared library path has been reset, the "
                "configured or default library file was not found, or the selected library failed "
                "to load.</p><p align=\"justify\">You may now either ";
            if (candownload)
                infotext += "download a pre-compiled SPARTA shared library file for your platform "
                            "from the SPARTA webserver, browse the ";
            else
                infotext += "browse the ";
            infotext += "filesystem for a suitable SPARTA library file, or exit SPARTA-GUI.</p>";
            mb.setInformativeText(infotext);

            QPushButton *downloadBtn = nullptr;
            if (candownload) {
                downloadBtn = mb.addButton("Download Library...", QMessageBox::ApplyRole);
                downloadBtn->setIcon(QIcon(":/icons/download-file.svg"));
            }
            auto *browseBtn = mb.addButton("Browse Filesystem...", QMessageBox::AcceptRole);
            browseBtn->setIcon(QIcon(":/icons/document-open.svg"));
            auto *exitBtn = mb.addButton("Exit", QMessageBox::NoRole);
            exitBtn->setIcon(QIcon(":/icons/application-exit.svg"));

            mb.setDefaultButton(candownload ? downloadBtn : browseBtn);
            mb.setEscapeButton(exitBtn);
            mb.exec();

            if (mb.clickedButton() == exitBtn) {
                // we cannot use QApplication::exit() here since we are still in the constructor
                exit(1);

            } else if (mb.clickedButton() == browseBtn) {
                QString pluginfile = QFileDialog::getOpenFileName(
                    this, "Select SPARTA shared library to use", ".", pattern, nullptr,
                    QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly);
                if (!pluginfile.isEmpty() && pluginfile.contains("libsparta", Qt::CaseSensitive)) {
                    auto canonical = QFileInfo(pluginfile).canonicalFilePath();
                    settings.setValue(Keys::PLUGIN_PATH, canonical);
                    settings.sync();
                    // must re-launch SPARTA-GUI to cleanly load the selected new plugin
                    relaunchApplication();
                    // This should not happen...
                    critical(this, "SPARTA-GUI Error", "Relaunching SPARTA-GUI failed.",
                             "SPARTA-GUI must be restarted to correctly load the selected "
                             "SPARTA shared library. Click on 'Close' to exit.");
                    exit(1);
                }
                // user cancelled file dialog -> loop back to show the dialog again

            } else if (mb.clickedButton() == downloadBtn) {
                // store in the same config directory where QSettings stores preferences
                const auto configDir =
                    QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
                if (configDir.isEmpty() || !QDir().mkpath(configDir)) {
                    critical(this, "SPARTA-GUI Error", "Cannot determine configuration directory.",
                             "Unable to create a writable directory in the user configuration "
                             "folder for storing the downloaded SPARTA shared library.");
                    continue;
                }
                auto libPath = configDir + QDir::separator() + libName;
                auto dlUrl   = getSpartaDownloadUrl();

                URLDownloader downloader(this);
                if (downloader.download(dlUrl, libPath, true)) {
                    // try loading the downloaded library
                    if (sparta.loadLib(libPath)) {
                        pluginPath = libPath;
                        settings.setValue(Keys::PLUGIN_PATH, pluginPath);
                        settings.sync();
                        // must re-launch SPARTA-GUI to cleanly load the selected new plugin
                        relaunchApplication();
                        // This should not happen...
                        critical(this, "SPARTA-GUI Error", "Relaunching SPARTA-GUI failed.",
                                 "SPARTA-GUI must be restarted to correctly load the selected "
                                 "SPARTA shared library. Click on 'Close' to exit.");
                        exit(1);
                    } else {
                        QFile::remove(libPath);
                        critical(this, "SPARTA-GUI Error",
                                 "Downloaded SPARTA library could not be loaded.",
                                 "<p align=\"justify\">The downloaded shared library file "
                                 "does not seem to be compatible with this system.</p>");
                    }
                } else {
                    critical(this, "SPARTA-GUI Error", "Failed to download SPARTA shared library.",
                             downloader.errorString());
                }
            }
        }
    }
}
#else
// dummy function when linking against library directly
void SpartaGui::setupPlugin(QSettings &) {}
#endif

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
    lastCpuBucket(-1), panels(nullptr), logwindow(nullptr), imagewindow(nullptr),
    chartwindow(nullptr), slideshow(nullptr), logupdater(nullptr), dirstatus(nullptr),
    progress(nullptr),
    prefdialog(nullptr), spartastatus(nullptr), varwindow(nullptr), runner(nullptr),
    runCounter(0), nthreads(1), mainx(width), mainy(height)
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
    setupAccelerators(settings);

    // populate the File->Open Example submenu (needs the plugin path for probing)
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

    // start SPARTA and initialize command completion
    startSparta();
    QStringList style_list;
    QFile internal_commands(":/sparta_internal_commands.txt");
    if (internal_commands.open(QIODevice::ReadOnly | QIODevice::Text)) {
        while (!internal_commands.atEnd()) {
            style_list << QString(internal_commands.readLine()).trimmed();
        }
    }
    internal_commands.close();
    int ncmds = sparta.styleCount("command");
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

    settings.beginGroup(Keys::GROUP_REFORMAT);
    textEdit->setReformatOnReturn(settings.value(Keys::RETURN, false).toBool());
    textEdit->setAutoComplete(settings.value(Keys::AUTOMATIC, true).toBool());
    autoLintEnabled = settings.value(Keys::AUTOLINT, true).toBool();
    settings.endGroup();

    // apply https proxy setting: prefer environment variable or fall back to preferences value
    applyProxySetting(sparta, settings);

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

    if (sparta.isRunning()) {
        stopRun();
        runner->wait();
        runner->deleteLater();
        runner = nullptr;
    }
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
}

void SpartaGui::showEditor()
{
    centralStack->setCurrentWidget(textEdit);
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

void SpartaGui::updateVariables()
{
    const auto doc = textEdit->toPlainText().replace('\t', ' ').split('\n');
    QStringList known;
    QRegularExpression indexvar(R"(^\s*variable\s+(\w+)\s+index\s+(.*))");
    QRegularExpression anyvar(R"(^\s*variable\s+(\w+)\s+(\w+)\s+(.*))");
    QRegularExpression usevar(R"((\$(\w)|\${(\w+)}))");
    QRegularExpression refvar(R"(v_(\w+))");

    // forget previously listed variables
    variables.clear();

    for (const auto &line : doc) {

        if (line.isEmpty()) continue;

        // first find variable definitions.
        // index variables are special since they can be overridden from the command line
        auto index = indexvar.match(line);
        auto any   = anyvar.match(line);

        if (index.hasMatch()) {
            if (index.lastCapturedIndex() >= 2) {
                auto name = index.captured(1);
                if (!known.contains(name)) {
                    variables.append(qMakePair(name, index.captured(2)));
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
}

// open file and switch CWD to path of file
void SpartaGui::openFile(const QString &fileName)
{
    // do nothing, if no file name provided
    if (fileName.isEmpty()) return;

    if (sparta.isRunning()) {
        stopRun();
        runner->wait();
        runner->deleteLater();
        runner = nullptr;
    }
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

    auto *viewer = new SlideShow(files.first());
    viewer->setAttribute(Qt::WA_DeleteOnClose);
    viewer->setWindowIcon(QIcon(Cfg::MAIN_ICON));
    viewer->show();

    // the import dialog of a movie file is modal to the (already visible)
    // slide show window, so a movie must not be added before it is shown
    for (const QString &f : files) {
        if (isMovieFile(f))
            viewer->addMovie(f);
        else
            viewer->addImage(f);
    }

    // every movie import was canceled or failed and no image was selected
    if (viewer->imageCount() == 0) viewer->close();
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
            auto *inspect_image = new ImageViewer(fileName, &sparta, this);
            inspect_image->setFont(font());
            inspect_image->setMinimumSize(Cfg::MINIMUM_WIDTH, Cfg::MINIMUM_HEIGHT);
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
    if (sparta.isRunning()) {
        stopRun();
        runner->wait();
        runner->deleteLater();
        runner = nullptr;
    }

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

void SpartaGui::stopRun()
{
    sparta.forceTimeout();
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

    // get timestep
    int step = 0;
    if (sparta.extractSetting("bigint") == 4)
        step = sparta.lastThermoAs<int>("step", 0);
    else
        step = static_cast<int>(sparta.lastThermoAs<int64_t>("step", 0));

    // extract cached stats data while SPARTA is executing a run command
    if (chartwindow && sparta.isRunning()) {
        // thermo data is not yet valid during setup
        if (sparta.lastThermoAs<int>("setup", 0)) return;

        sparta.lastThermo("lock", 0);
        const int ncols = sparta.lastThermoAs<int>("num", 0);
        if (ncols > 0) updateChartData(step, ncols);
        sparta.lastThermo("unlock", 0);
    }

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

void SpartaGui::updateSlideShow()
{
    // update list of available image file names
    QString imagefile = sparta.lastThermoString("imagename", 0);
    if (imagefile.isEmpty()) return;

    if (!slideshow) {
        slideshow = new SlideShow(currentFile, this);
        panels->setPanelWidget(PanelManager::Slide, slideshow,
                               QString("Slide Show - %1 - Run %2").arg(currentFile).arg(runCounter));
        if (QSettings().value(Keys::VIEWSLIDE, true).toBool())
            panels->openPanel(PanelManager::Slide);
        else
            panels->closePanel(PanelManager::Slide);
    } else {
        slideshow->setWindowTitle(
            QString("SPARTA-GUI - Slide Show - %1 - Run %2").arg(currentFile).arg(runCounter));
        if (QSettings().value(Keys::VIEWSLIDE, true).toBool()) panels->openPanel(PanelManager::Slide);
    }
    slideshow->addImage(imagefile);
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
    progress->setValue(Cfg::PROGRESS_MAXIMUM);
    textEdit->setHighlight(CodeEditor::NO_HIGHLIGHT, false);

    capturer->endCapture();

    if (logwindow) {
        auto log = capturer->getCapture();
        logwindow->insertPlainText(log.c_str());
        logwindow->moveCursor(QTextCursor::End);
    }

    warnHighBufferUsage();

    finalizeChartData();

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
    if (sparta.isRunning()) {
        warning(this, "SPARTA-GUI Warning", "Must stop current run before relaunching SPARTA");
        return;
    }
    {
        StdoutSilencer guard;
        sparta.close();
    }
}

void SpartaGui::createLogWindow(QSettings &settings)
{
    logwindow = new LogWindow(currentFile, this);
    logwindow->setReadOnly(true);
    logwindow->setCenterOnScroll(true);
    logwindow->moveCursor(QTextCursor::End);
    logwindow->setLineWrapMode(LogWindow::NoWrap);

    const bool keepOld = !settings.value(Keys::LOGREPLACE, true).toBool();
    panels->setPanelWidget(PanelManager::Log, logwindow,
                          QString("Output - %1 - Run %2").arg(currentFile).arg(runCounter),
                          keepOld);

    if (settings.value(Keys::VIEWLOG, true).toBool())
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

    if (settings.value(Keys::VIEWCHART, true).toBool())
        panels->openPanel(PanelManager::Chart);
    else
        panels->closePanel(PanelManager::Chart);
}

void SpartaGui::doRun(bool use_buffer)
{
    if (sparta.isRunning()) {
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

    runner = new SpartaRunner(this);
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
    if (use_buffer) {
        // always add final newline since the text edit widget does not do it
        runner->setupRun(&sparta, (textEdit->toPlainText() + "\n").toStdString());
    } else {
        runner->setupRun(&sparta, {}, currentFile.toStdString());
    }

    // apply https proxy setting: prefer environment variable or fall back to preferences value
    applyProxySetting(sparta, settings);

    connect(runner, &SpartaRunner::resultReady, this, &SpartaGui::runDone);
    connect(runner, &SpartaRunner::finished, runner, &QObject::deleteLater);
    runner->start();

    createLogWindow(settings);

    createChartWindow(settings);

    if (slideshow) {
        slideshow->setWindowTitle(QString("SPARTA-GUI - Slide Show - " + currentFile));
        slideshow->clear();
        panels->closePanel(PanelManager::Slide);
    }

    logupdater = new QTimer(this);
    connect(logupdater, &QTimer::timeout, this, &SpartaGui::logUpdate);
    logupdater->start(settings.value(Keys::UPDFREQ, Cfg::DATA_UPDATE_INTERVAL_DEFAULT).toInt());
}

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
    panels->setPanelWidget(PanelManager::ProjectFiles, projectFilesList, "Project Files");
}

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
        item->setIcon(QIcon(":/icons/document-open.svg"));
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
    if (slideshow) images = slideshow->images();
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
        "Surface geometry (*.stl *.surf);;STL files (*.stl);;"
        "SPARTA surface files (*.surf);;All files (*)");
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

void SpartaGui::renderImage()
{
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
                    warning(this, "Image Viewer File Creation Error",
                            "SPARTA failed to create the image:",
                            QString("<br><code>%1</code>").arg(errmsg));
                    return;
                }
            }
            textEdit->setTextCursor(saved);
            // still no system box. bail out with a suitable message
            if (!sparta.extractSetting("box_exist")) {
                warning(this, "Image Viewer File Creation Error",
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

        imagewindow = new ImageViewer(currentFile, &sparta, this);
        const bool keepOld = !QSettings().value(Keys::IMAGEREPLACE, true).toBool();
        panels->setPanelWidget(PanelManager::Image, imagewindow,
                               QString("Image - %1").arg(currentFile), keepOld);
    } else {
        warning(this, "Image Viewer File Creation Error",
                "Cannot create snapshot image while SPARTA is running");
        return;
    }
    panels->openPanel(PanelManager::Image);
}

#if defined(SPARTA_GUI_HAVE_VTK)
void SpartaGui::open3DViewer()
{
    if (!vtkViewer) vtkViewer = new VtkViewer(this);
    vtkViewer->showViewer();
}

void SpartaGui::renderVtkSnapshot()
{
    if (sparta.isRunning()) {
        warning(this, "3D Snapshot", "Cannot create a 3D snapshot while SPARTA is running.");
        return;
    }
    if (!vtkViewer) vtkViewer = new VtkViewer(this);
    startSparta();

    // does the loaded library actually provide the VTK dump styles?  (VTK is an
    // optional SPARTA package; a stock build will not have them.)  If not, just
    // open the viewer so the user can still load dump-vtk files written elsewhere.
    bool haveVtkDump = false;
    const int ndumpstyles = sparta.styleCount("dump");
    for (int i = 0; i < ndumpstyles; ++i)
        if (sparta.styleName("dump", i).endsWith("/vtk")) { haveVtkDump = true; break; }
    if (!haveVtkDump) {
        vtkViewer->showViewer();
        warning(this, "3D Snapshot",
                "This SPARTA library was built without the VTK package,",
                "so it cannot write VTK files directly.  You can still open <code>.vtu</code> / "
                "<code>.vtp</code> files written by a VTK-enabled SPARTA build (or the "
                "<i>Export to ParaView</i> tools) with the viewer's <b>Open</b> button.");
        return;
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
            warning(this, "3D Snapshot",
                    "Cannot create a 3D snapshot from an input that does not create a system box.");
            return;
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
        VtkViewer::Kind kind;
    };
    const QList<Cat> cats = {
        {"sgvtkgrid", "grid/vtk", "vtu", "all id proc", "grid", VtkViewer::Kind::Grid},
        {"sgvtkpart", "particle/vtk", "vtp", "all id x y z vx vy vz", "particles",
         VtkViewer::Kind::Particles},
        {"sgvtksurf", "surf/vtk", "vtp", "all id type", "surfaces", VtkViewer::Kind::Surface},
    };

    vtkViewer->clearScene();
    int loaded = 0;
    for (const auto &c : cats) {
        const QString f = renderCategory(c.id, c.style, c.ext, c.attrs);
        if (f.isEmpty()) continue;
        if (vtkViewer->addDataset(f, c.label, c.kind, nullptr)) ++loaded;
        QFile::remove(f);
    }

    vtkViewer->showViewer();
    if (loaded == 0)
        warning(this, "3D Snapshot",
                "No particle, grid or surface data was produced for the current state.");
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
    slideshow        = nullptr;
    imagewindow      = nullptr;
    varwindow        = nullptr;
    diagnosticsList  = nullptr;
    projectFilesList = nullptr;
    sweepPanel       = nullptr;
    historyPanel     = nullptr;
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
    const auto libName   = getSpartaLibName();
    const auto configDir = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    auto libPath         = configDir + QDir::separator() + libName;
    auto dlUrl           = getSpartaDownloadUrl();

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
            if (downloader.download(dlUrl, libPath, true)) {
                warning(this, "SPARTA Shared Library Updated",
                        "The latest SPARTA library has been downloaded successfully. "
                        "SPARTA-GUI must be relaunched to activate it.");
                relaunchApplication();
            } else {
                critical(this, "Check for SPARTA Update",
                         "Failed to download SPARTA shared library.", downloader.errorString());
            }
        }
        return;
    }
}
#endif

void SpartaGui::help()
{
    QMessageBox mb(this);
    mb.setWindowTitle("SPARTA-GUI Quick Help");
    mb.setWindowIcon(QIcon(Cfg::MAIN_ICON));
    mb.setText("<div>This is SPARTA-GUI version " SPARTA_GUI_VERSION "</div>");
    mb.setInformativeText(
        "<p>SPARTA-GUI is a graphical text editor that is customized for "
        "editing SPARTA input files and linked to the SPARTA "
        "library and thus can run SPARTA directly using the contents of the "
        "text buffer as input. It can retrieve and display information from "
        "SPARTA while it is running and display visualizations created "
        "with the dump image command.</p>"
        "<p>The main window of the SPARTA-GUI is a text editor window with "
        "SPARTA specific syntax highlighting. When typing <b>Ctrl-Enter</b> "
        "or clicking on 'Run SPARTA from Editor Buffer' in the 'Run' menu, "
        "SPARTA will be run "
        "with the contents of editor buffer as input. The output of the SPARTA "
        "run is captured and displayed in an Output window. The stats output data "
        "is displayed in a chart window. Both are updated regularly during the "
        "run, as is a progress bar in the main window. The running simulation "
        "can be stopped cleanly by typing <b>Ctrl-/</b> or by clicking on "
        "'Stop SPARTA' in the 'Run' menu. While SPARTA is not running, "
        "an image of the simulated system can be created and shown in an image "
        "viewer window by typing <b>Ctrl-i</b> or by clicking on 'Create Image' "
        "in the 'Run' menu. Multiple image settings can be changed through the "
        "buttons in the menu bar and the image will be re-rendered. In case "
        "an input file contains a dump image command, SPARTA-GUI will load "
        "the images as they are created and display them in a slide show. </p>"
        "<p>When opening a file, the editor will determine the directory "
        "where the input file resides and switch its current working directory "
        "to that same folder and thus enabling the run to read other files in "
        "that folder, e.g. a surface or grid file. The GUI will show its current working "
        "directory in the status bar. In addition to using the menu, the "
        "editor window can also receive files as the first command line "
        "argument or via drag-n-drop from a graphical file manager or a "
        "desktop environment.</p>"
        "<p>Almost all commands are accessible via keyboard shortcuts. Which "
        "those shortcuts are, is typically shown next to their entries in the "
        "menus. "
        "In addition, the documentation for the command in the current line "
        "can be viewed by typing <b>Ctrl-?</b> or by choosing the respective "
        "entry in the context menu, available by right-clicking the mouse. "
        "Log, chart, slide show, and image windows can be closed with "
        "<b>Ctrl-W</b> and the application terminated with <b>Ctrl-Q</b>.</p>"
        "<p>The 'About SPARTA-GUI' dialog will show the SPARTA version and the "
        "features included into the SPARTA library linked to the SPARTA-GUI. "
        "A number of settings can be adjusted in the 'Preferences' dialog (in "
        "the 'Edit' menu or from <b>Ctrl-P</b>) which includes selecting "
        "the KOKKOS accelerator package and number of OpenMP threads. Due to its nature "
        "as a graphical application, it is <b>not</b> possible to use the "
        "SPARTA-GUI in parallel with MPI.</p>");
    mb.setIconPixmap(QPixmap(Cfg::MAIN_ICON).scaled(64, 64));
    mb.setStandardButtons(QMessageBox::Close);
    auto *button = mb.button(QMessageBox::Close);
    button->setIcon(QIcon(":/icons/window-close.svg"));
    mb.setFont(font());
    mb.exec();
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
    QSettings settings;
    settings.clear();
    settings.sync();
}

void SpartaGui::editVariables()
{
    QList<QPair<QString, QString>> newvars = variables;
    SetVariables vars(newvars);
    vars.setFont(font());
    if (vars.exec() == QDialog::Accepted) {
        variables = newvars;
        if (sparta.isRunning()) {
            stopRun();
            runner->wait();
            runner->deleteLater();
            runner = nullptr;
        }
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
            if (sparta.isRunning()) {
                stopRun();
                runner->wait();
                runner->deleteLater();
                runner = nullptr;
            }
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
        if (imagewindow && !instanceClosed) imagewindow->createImage();
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
