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

// The main window itself.
//
// Nothing constructed SpartaGui outside of main(), so the largest source file
// in the application had no test at all: the menus, their shortcuts, and which
// panels each workspace comes up with were only ever checked by driving the
// running program under Xvfb, which takes minutes and needs a display.
//
// This runs offscreen in under a second. It deliberately drives only what is
// safe to trigger unattended -- anything that opens a modal file dialog, writes
// a file, or starts a run would block or leave debris, so those actions are
// checked for existence, shortcut and icon rather than triggered.

#include "actionscan.h"

#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QFont>
#include <QIcon>
#include <QKeySequence>
#include <QMenu>
#include <QMenuBar>
#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>

#include <DockAreaWidget.h>
#include <DockManager.h>
#include <DockWidgetTab.h>
#include <DockWidget.h>

#include <map>
#include <memory>
#include <set>
#include <string>

#include "dockpanels.h"
#include "constants.h"
#include "helpers.h"
#include "preferences.h"
#include "slideshow.h"
#include "spartagui.h"

using ads::CDockWidget;

namespace {

// Names of the actions that must not be triggered unattended: each opens a
// modal dialog that would sit there with nobody to dismiss it, writes to the
// filesystem, or starts a simulation.
const std::set<QString> BLOCKING = {
    "&New Input File",
    "&Open Input File",
    "&Save Input File",
    "Save Input File &As",
    "&View Text File",
    "View &Image or Movie File(s)...",
    "&Plot Data File...",
    "Inspect &Restart File",
    "&Quit",
    "&Find and Replace...",
    "Insert &Snippet...",
    "P&references...",
    "Reset Preferences to &Defaults",
    "&Run SPARTA from Editor Buffer",
    "Run SPARTA from &File",
    "&Stop SPARTA",
    "Chec&k Input",
    "Relaunch &SPARTA Instance",
    "Set &Variables...",
    "Insert &Restart Commands...",
    "Create &Image",
    "3D &Snapshot (VTK)",
    "Import Sur&face (STL / SPARTA)...",
    "Export to Para&View...",
    "Surface &Quantities Report...",
    "Parametric S&weep...",
    "Run &History...",
    "&About SPARTA-GUI",
    "Quick &Help",
    "SPARTA-&GUI Documentation",
    "SPARTA Online &Manual",
    "Check for &SPARTA update",
};

// Where to find a shared libsparta. Baked in by the build; the environment
// still wins so the binary can be run by hand against another one.
const char *testLibrary()
{
    static const QByteArray env = qgetenv("SPARTA_PLUGIN_LIB");
    if (!env.isEmpty()) return env.constData();
#if defined(SPARTA_TEST_LIBRARY_PATH)
    return SPARTA_TEST_LIBRARY_PATH;
#else
    return "";
#endif
}

#define REQUIRE_LIBRARY()                                                                  \
    do {                                                                                   \
        if (!*testLibrary())                                                               \
            GTEST_SKIP() << "no shared libsparta: configure with -D SPARTA_TEST_LIBRARY="; \
    } while (0)

// A window per test, on a settings scope of its own, so one test's preferences
// cannot decide another's outcome.
class MainWindow : public ::testing::Test {
protected:
    void SetUp() override
    {
        REQUIRE_LIBRARY();
        QSettings settings;
        settings.clear();
        // Without this the constructor puts up its "No SPARTA Shared Library"
        // box and loops on it until someone picks a library or exits -- which
        // offscreen, with nobody to click, is a hang rather than a failure.
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.sync();

        // A modal must never be allowed to stop this suite. The window's
        // constructor loops on its "No SPARTA Shared Library" box until someone
        // answers, and offscreen nobody can -- so a settings problem shows up
        // as a test that never returns rather than one that fails. The reaper
        // closes anything modal and the check below turns it into a failure.
        modalSeen = false;
        QObject::connect(&reaper, &QTimer::timeout, [this]() {
            if (QWidget *m = QApplication::activeModalWidget()) {
                modalSeen = true;
                m->close();
            }
        });
        reaper.start(50);

        gui = new SpartaGui(nullptr, QString(), 800, 600);
        reaper.stop();
        ASSERT_FALSE(modalSeen)
            << "the main window put up a modal dialog while being constructed; with no library "
               "configured it loops on one until answered";
    }
    void TearDown() override
    {
        delete gui;
        gui = nullptr;
        QSettings().clear();
    }

    // Walk the menu bar rather than findChildren<QAction*>(). Qt-ADS drops a
    // dock widget out of the window's object tree when it is hidden inside a
    // tabbed area, taking its toggle action with it -- four of the eight panel
    // entries were invisible to findChildren even though the View menu shows
    // all eight. The menu is also what a user actually has in front of them.
    static void collect(QMenu *menu, QList<QAction *> &out)
    {
        for (auto *a : menu->actions()) {
            if (a->isSeparator()) continue;
            out.append(a);
            if (a->menu()) collect(a->menu(), out);
        }
    }

    QList<QAction *> allActions() const
    {
        QList<QAction *> out;
        auto *bar = gui->findChild<QMenuBar *>();
        if (bar)
            for (auto *top : bar->actions()) {
                out.append(top);
                if (top->menu()) collect(top->menu(), out);
            }
        return out;
    }

    QAction *action(const QString &text) const
    {
        for (auto *a : allActions())
            if (a->text() == text) return a;
        return nullptr;
    }

    // Asked of the dock manager by object name rather than of PanelManager, so
    // this needs no access to a private member. findChild() is not enough: a
    // dock hidden inside a tabbed area is not in the window's object tree, and
    // four of the eight panels are exactly that.
    bool panelOpen(const char *objectName) const
    {
        auto *dm = gui->findChild<ads::CDockManager *>();
        if (!dm) return false;
        CDockWidget *d = dm->findDockWidget(objectName);
        EXPECT_NE(d, nullptr) << objectName << " does not exist";
        return d && !d->isClosed();
    }

    void enterWorkspace(const QString &name)
    {
        QAction *a = action(name);
        ASSERT_NE(a, nullptr) << name.toStdString() << " is missing";
        a->trigger();
        QCoreApplication::processEvents();
    }

    SpartaGui *gui = nullptr;
    QTimer reaper;
    bool modalSeen = false;
};

} // namespace

// ---------------------------------------------------------------- inventory

TEST_F(MainWindow, HasTheSixTopLevelMenus)
{
    auto *bar = gui->findChild<QMenuBar *>();
    ASSERT_NE(bar, nullptr);

    QStringList titles;
    for (auto *a : bar->actions())
        if (a->menu()) titles << a->text();

    EXPECT_EQ(titles, (QStringList{"&File", "&Edit", "&Run", "&Tools", "&View", "&About"}));
}

TEST_F(MainWindow, EveryMenuActionHasTextAndAnIcon)
{
    auto *bar = gui->findChild<QMenuBar *>();
    ASSERT_NE(bar, nullptr);

    int checked = 0;
    for (auto *top : bar->actions()) {
        QMenu *menu = top->menu();
        if (!menu) continue;
        for (auto *a : menu->actions()) {
            if (a->isSeparator()) continue;
            EXPECT_FALSE(a->text().isEmpty())
                << "an entry of " << top->text().toStdString() << " has no text";
            // The recent-file slots are numbered placeholders until a file has
            // been opened, and the panel toggles take the dock's own icon.
            if (a->text().size() > 3)
                EXPECT_FALSE(a->icon().isNull())
                    << a->text().toStdString() << " has no icon";
            ++checked;
        }
    }
    EXPECT_GT(checked, 40) << "only " << checked << " menu entries found; the menus look truncated";
}

TEST_F(MainWindow, NoTwoActionsShareAShortcut)
{
    std::map<QString, QString> seen;
    for (auto *a : allActions()) {
        const QKeySequence key = a->shortcut();
        if (key.isEmpty()) continue;
        const QString text = key.toString();
        auto it = seen.find(text);
        if (it != seen.end())
            ADD_FAILURE() << text.toStdString() << " is bound to both \""
                          << it->second.toStdString() << "\" and \"" << a->text().toStdString()
                          << "\"";
        else
            seen.emplace(text, a->text());
    }
    EXPECT_GT(seen.size(), 25u) << "only " << seen.size()
                                << " shortcuts found; the menus look truncated";
}

TEST_F(MainWindow, TheDocumentedShortcutsAreBound)
{
    // Every shortcut the manual lists. A rename that drops one of these breaks
    // muscle memory silently.
    const std::map<QString, QString> expected = {
        {"&New Input File", "Ctrl+N"},        {"&Open Input File", "Ctrl+O"},
        {"&Welcome Screen", "Alt+Home"},
        {"&Save Input File", "Ctrl+S"},       {"Save Input File &As", "Ctrl+Shift+S"},
        {"&View Text File", "Ctrl+Shift+F"},  {"&Plot Data File...", "Ctrl+Shift+P"},
        {"Inspect &Restart File", "Ctrl+Shift+R"},
        {"View &Image or Movie File(s)...", "Ctrl+Shift+J"},
        {"&Quit", "Ctrl+Q"},                  {"&Undo", "Ctrl+Z"},
        {"&Redo", "Ctrl+Shift+Z"},            {"&Copy", "Ctrl+C"},
        {"Cu&t", "Ctrl+X"},                   {"&Paste", "Ctrl+V"},
        {"&Find and Replace...", "Ctrl+F"},   {"P&references...", "Ctrl+P"},
        {"Chec&k Input", "Ctrl+K"},           {"Set &Variables...", "Ctrl+Shift+V"},
        {"Create &Image", "Ctrl+I"},
        {"&Run SPARTA from Editor Buffer", "Ctrl+Return"},
        {"Run SPARTA from &File", "Ctrl+Shift+Return"},
        {"&Stop SPARTA", "Ctrl+/"},           {"Slide S&how in Viewer", "Ctrl+L"},
        {"&Run Workspace", "Ctrl+1"},         {"&Analyze Workspace", "Ctrl+2"},
        {"&Visualize Workspace", "Ctrl+3"},
        {"&Output Window", "Ctrl+Shift+L"},   {"&Charts Window", "Ctrl+Shift+C"},
        {"&Viewer Window", "Ctrl+Shift+I"},   {"&Variables Window", "Ctrl+Shift+W"},
        {"Import Sur&face (STL / SPARTA)...", "Ctrl+Shift+T"},
        {"Export to Para&View...", "Ctrl+Shift+E"},
        {"&About SPARTA-GUI", "Ctrl+Shift+A"},
        {"Quick &Help", "Ctrl+Shift+H"},
        {"SPARTA-&GUI Documentation", "Ctrl+Shift+G"},
        {"SPARTA Online &Manual", "Ctrl+Shift+M"},
        {"Check for &SPARTA update", "Ctrl+Shift+U"},
    };

    for (const auto &[text, key] : expected) {
        QAction *a = action(text);
        ASSERT_NE(a, nullptr) << text.toStdString() << " is gone";
        EXPECT_EQ(a->shortcut(), QKeySequence(key))
            << text.toStdString() << " is bound to \"" << a->shortcut().toString().toStdString()
            << "\" rather than " << key.toStdString();
    }

    // The 3D entries exist only where the viewer was built, and a build without
    // VTK is a supported configuration -- asserting them unconditionally makes
    // this suite fail there for a reason that is not a defect.
#if defined(SPARTA_GUI_HAVE_VTK)
    QAction *snap = action("3D &Snapshot (VTK)");
    ASSERT_NE(snap, nullptr) << "built with VTK but the 3D snapshot entry is gone";
    EXPECT_EQ(snap->shortcut(), QKeySequence("Ctrl+Shift+3"));
#else
    EXPECT_EQ(action("3D &Snapshot (VTK)"), nullptr)
        << "built without VTK, yet the menu offers a 3D snapshot";
    EXPECT_EQ(action("3D Viewer &Window (VTK)"), nullptr)
        << "built without VTK, yet the menu offers the 3D viewer window";
#endif
}

// ---------------------------------------------------------------- workspaces

TEST_F(MainWindow, RunAddsTheVariablesToTheEditorAndItsOutput)
{
    enterWorkspace("&Run Workspace");

    EXPECT_TRUE(panelOpen("dockOutput"))
        << "Run came up with no output panel. It is created lazily by a run, and a workspace "
           "only opens a panel that already holds a widget, so without ensureLogPanel() this "
           "mode is a bare editor until the user runs something.";
    EXPECT_TRUE(panelOpen("dockVariables"));
    EXPECT_FALSE(panelOpen("dockCharts")) << "Run opened the charts, squeezing the deck";
    EXPECT_FALSE(panelOpen("dockViewer")) << "Run opened the viewer, squeezing the deck";
    EXPECT_FALSE(panelOpen("dockDiagnostics")) << "Run opened the linter";
    EXPECT_FALSE(panelOpen("dockProjectFiles")) << "Run opened the file navigator";
}

TEST_F(MainWindow, AnalyzeShowsThePlotsBeforeAnythingHasRun)
{
    enterWorkspace("&Analyze Workspace");

    // The chart panel is built lazily by a run, and a workspace only opens a
    // panel that already holds a widget. Before this was arranged, selecting
    // Analyze on a freshly opened deck showed the deck and nothing else.
    EXPECT_TRUE(panelOpen("dockCharts")) << "Analyze came up with no chart panel";
    EXPECT_FALSE(panelOpen("dockViewer"))
        << "Analyze opened the viewer; the whole window is meant to be the plots";
    EXPECT_FALSE(panelOpen("dockOutput")) << "Analyze opened the console output";
}

TEST_F(MainWindow, VisualizeShowsThePicturesBeforeAnythingHasRun)
{
    enterWorkspace("&Visualize Workspace");

    EXPECT_TRUE(panelOpen("dockViewer")) << "Visualize came up with no viewer panel";
    EXPECT_FALSE(panelOpen("dockCharts"))
        << "Visualize opened the charts; the whole window is meant to be the pictures";
}

TEST_F(MainWindow, EveryWorkspaceOpensExactlyWhatItDocuments)
{
    const struct {
        const char *action;
        PanelManager::Mode mode;
    } modes[] = {
        {"&Run Workspace", PanelManager::RunMode},
        {"&Analyze Workspace", PanelManager::Analyze},
        {"&Visualize Workspace", PanelManager::Visualize},
    };
    const struct {
        const char *dock;
        PanelManager::Panel panel;
    } panels[] = {
        {"dockOutput", PanelManager::Log},        {"dockCharts", PanelManager::Chart},
        {"dockViewer", PanelManager::Viewer},     {"dockVariables", PanelManager::Variables},
        {"dockSweep", PanelManager::Sweep},       {"dockHistory", PanelManager::History},
        {"dockDiagnostics", PanelManager::Diagnostics},
        {"dockProjectFiles", PanelManager::ProjectFiles},
    };

    for (const auto &m : modes) {
        enterWorkspace(m.action);
        for (const auto &p : panels)
            EXPECT_EQ(panelOpen(p.dock), PanelManager::modeShows(m.mode, p.panel))
                << PanelManager::panelName(p.panel).toStdString() << " in the "
                << PanelManager::modeName(m.mode).toStdString() << " workspace";
    }
}

// ---------------------------------------------------------------- panel menu

TEST_F(MainWindow, EveryPanelCanBeOpenedFromTheViewMenu)
{
    // The View menu names the panels as windows; each entry has to build its
    // content on demand, because nothing else will have before the first run.
    const struct {
        const char *entry;
        const char *dock;
    } entries[] = {
        {"&Output Window", "dockOutput"},
        {"&Charts Window", "dockCharts"},
        {"&Viewer Window", "dockViewer"},
        {"&Variables Window", "dockVariables"},
        {"Parametric S&weep Window", "dockSweep"},
        {"Run &History Window", "dockHistory"},
        {"&Diagnostics Window", "dockDiagnostics"},
        {"Project &Files Window", "dockProjectFiles"},
    };

    for (const auto &e : entries) {
        QAction *a = action(e.entry);
        ASSERT_NE(a, nullptr) << e.entry << " is missing from the View menu";
        if (!a->isChecked()) {
            a->trigger();
            QCoreApplication::processEvents();
        }
        EXPECT_TRUE(panelOpen(e.dock))
            << e.entry << " left its panel closed, which is what happens when the panel has no "
                          "widget yet and nothing creates one on demand";

        a->trigger(); // and back, so the next entry starts from a known state
        QCoreApplication::processEvents();
        EXPECT_FALSE(panelOpen(e.dock)) << e.entry << " could not close its panel again";
    }
}

TEST_F(MainWindow, ThePanelMenuEntriesKeepTheirNamesWhenThePanelIsRetitled)
{
    // The dock's title and its menu entry are different strings: the title
    // carries the file and run number, the menu entry stays "Output Window".
    // Qt-ADS overwrites the entry from the title unless that is undone.
    QAction *entry = action("&Output Window");
    ASSERT_NE(entry, nullptr);

    // Asked of the dock manager, not findChild(): a dock that is closed or is
    // hidden inside a tabbed area is not in the window's object tree, which is
    // the same reason panelOpen() above goes through the manager.
    auto *dm = gui->findChild<ads::CDockManager *>();
    ASSERT_NE(dm, nullptr);
    auto *dock = dm->findDockWidget("dockOutput");
    ASSERT_NE(dock, nullptr);
    dock->setWindowTitle("Output - in.circle - Run 3");
    QCoreApplication::processEvents();

    EXPECT_EQ(entry->text(), QString("&Output Window"))
        << "the View menu now reads \"" << entry->text().toStdString() << "\"";
}

TEST_F(MainWindow, EveryPanelDockIsTitled)
{
    // A dock with an empty title shows a blank tab: the panel is on screen with
    // nothing naming it, which is how Project Files looked.
    auto *dm = gui->findChild<ads::CDockManager *>();
    ASSERT_NE(dm, nullptr);

    for (const char *name : {"dockOutput", "dockCharts", "dockViewer", "dockVariables",
                             "dockSweep", "dockHistory", "dockDiagnostics", "dockProjectFiles"}) {
        CDockWidget *d = dm->findDockWidget(name);
        ASSERT_NE(d, nullptr) << name;
        EXPECT_FALSE(d->windowTitle().trimmed().isEmpty()) << name << " has a blank tab";
    }

    // and again once each has been opened and filled in on demand
    for (const auto &e : {std::make_pair("&Output Window", "dockOutput"),
                          std::make_pair("&Charts Window", "dockCharts"),
                          std::make_pair("&Variables Window", "dockVariables"),
                          std::make_pair("Parametric S&weep Window", "dockSweep"),
                          std::make_pair("Run &History Window", "dockHistory"),
                          std::make_pair("&Diagnostics Window", "dockDiagnostics"),
                          std::make_pair("Project &Files Window", "dockProjectFiles")}) {
        QAction *a = action(e.first);
        ASSERT_NE(a, nullptr) << e.first;
        if (!a->isChecked()) a->trigger();
        QCoreApplication::processEvents();
        CDockWidget *d = dm->findDockWidget(e.second);
        ASSERT_NE(d, nullptr) << e.second;
        EXPECT_FALSE(d->windowTitle().trimmed().isEmpty())
            << e.second << " has a blank tab after being opened";

        // The tab has to carry that title, since the tab is what names the
        // panel on screen. Whether it is *shown* cannot be settled here -- the
        // window is never mapped, so every tab reports itself hidden -- and is
        // checked by the screenshot sweep instead.
        auto *tab = d->tabWidget();
        ASSERT_NE(tab, nullptr) << e.second << " has no tab";
        EXPECT_EQ(tab->text(), d->windowTitle()) << e.second;
    }
}

TEST_F(MainWindow, TheViewerMenuEntriesEachBringUpTheirPage)
{
    // Each of these picks which page of the viewer is in front, and each page
    // is built lazily by whatever produces its content -- a render, a run
    // writing dump images. An entry whose page does not exist yet used to do
    // nothing whatsoever: on a deck with its dumps commented out, which is
    // most of the examples, Slide Show in Viewer was a no-op with no
    // explanation.
    auto *dm = gui->findChild<ads::CDockManager *>();
    ASSERT_NE(dm, nullptr);

    for (const char *entry : {"S&napshot in Viewer", "Slide S&how in Viewer"}) {
        QAction *a = action(entry);
        ASSERT_NE(a, nullptr) << entry << " is missing";
        a->trigger();
        QCoreApplication::processEvents();

        CDockWidget *d = dm->findDockWidget("dockViewer");
        ASSERT_NE(d, nullptr);
        EXPECT_FALSE(d->isClosed()) << entry << " left the viewer panel closed";
    }

    // and the frame view is really there, not merely asked for
    EXPECT_NE(gui->findChild<SlideShow *>(), nullptr)
        << "Slide Show in Viewer did not build a frame view";
}

// ---------------------------------------------------------------- behaviour

TEST_F(MainWindow, StopIsOnlyOfferedWhileSomethingIsRunning)
{
    // Stop was always enabled. Nothing is running here, so picking it did
    // nothing whatsoever -- forceTimeout() on an idle instance is a no-op --
    // and a control that can always be picked and never does anything is
    // indistinguishable from one that is broken.
    QAction *stop = action("&Stop SPARTA");
    ASSERT_NE(stop, nullptr);
    EXPECT_FALSE(stop->isEnabled())
        << "Stop SPARTA is offered with no run in progress";

    // and the status bar button beside it agrees
    for (auto *b : gui->findChildren<QPushButton *>())
        if (b->toolTip() == "Stop SPARTA")
            EXPECT_FALSE(b->isEnabled()) << "the stop button is offered with no run in progress";

    // the things that start a run stay available
    for (const char *entry : {"&Run SPARTA from Editor Buffer", "Run SPARTA from &File"}) {
        QAction *a = action(entry);
        ASSERT_NE(a, nullptr) << entry;
        EXPECT_TRUE(a->isEnabled()) << entry << " is greyed out with nothing running";
    }
}

TEST_F(MainWindow, EnteringAWorkspaceNeverPutsUpADialog)
{
    // Entering Analyze or Visualize opens the viewer, and opening the viewer
    // renders so the pane is not blank. On a buffer that cannot be rendered --
    // an empty one, or a deck that never creates a box, which is the state
    // here -- that render used to answer a workspace switch with a modal error
    // box. Offscreen that is not a dialog anyone can dismiss: it is a hang.
    for (const char *entry : {"&Run Workspace", "&Analyze Workspace", "&Visualize Workspace"}) {
        enterWorkspace(entry);
        for (auto *w : QApplication::topLevelWidgets()) {
            auto *dlg = qobject_cast<QDialog *>(w);
            if (dlg && dlg->isVisible())
                ADD_FAILURE() << entry << " put up \"" << dlg->windowTitle().toStdString()
                              << "\"";
        }
    }
    EXPECT_EQ(QApplication::activeModalWidget(), nullptr)
        << "a workspace switch left a modal dialog on screen";
}

// The snapshot preferences came across from LAMMPS carrying VDW Style,
// Dynamic Bonds and Bond Cutoff. SPARTA has particles, grid cells and surface
// elements -- no bonds, no van der Waals radii -- and all three only ever wrote
// a settings key nothing read.
TEST_F(MainWindow, TheSnapshotPreferencesOfferNothingAboutBonds)
{
    QSettings settings;
    SnapshotTab tab(&settings);
    for (const char *gone : {"vdwstyle", "autobond", "bondcut"})
        EXPECT_EQ(tab.findChild<QWidget *>(gone), nullptr)
            << gone << " is back in the snapshot preferences";

    for (auto *label : tab.findChildren<QLabel *>()) {
        const QString t = label->text();
        EXPECT_FALSE(t.contains("Bond", Qt::CaseInsensitive))
            << "a preferences label still reads \"" << t.toStdString() << "\"";
        EXPECT_FALSE(t.contains("VDW", Qt::CaseInsensitive))
            << "a preferences label still reads \"" << t.toStdString() << "\"";
    }
}

TEST_F(MainWindow, TheNonBlockingActionsCanAllBeTriggeredWithoutCrashing)
{
    int triggered = 0;
    for (auto *a : allActions()) {
        if (a->text().isEmpty() || BLOCKING.count(a->text())) continue;
        if (!a->isEnabled()) continue;
        a->trigger();
        QCoreApplication::processEvents();
        ++triggered;
    }
    EXPECT_GT(triggered, 15) << "only " << triggered
                             << " actions were safe to trigger; the menus look truncated";
    // Getting here at all is the assertion: any of these taking the window down
    // fails the test by killing the process.
    SUCCEED();
}

TEST_F(MainWindow, TheWindowKeepsTheSizeItWasAskedFor)
{
    EXPECT_EQ(gui->width(), 800);
    EXPECT_EQ(gui->height(), 600);
}

// ------------------------------------------------------------ discoverability

// The status-tip table lives apart from the ~70 addMenuAction() call sites, so
// nothing structural keeps it complete: an action added without a table entry
// compiles and runs, it just hovers silent. This test is the enforcement --
// the table stays complete because forgetting it fails here, by name.
TEST_F(MainWindow, EveryMenuActionCarriesAStatusTip)
{
    const auto infos = scanMenuBar(gui->findChild<QMenuBar *>());
    ASSERT_GT(infos.size(), 40) << "the walker lost most of the menu tree";

    QStringList silent;
    for (const auto &info : infos)
        if (info.action && info.action->statusTip().isEmpty())
            silent << (info.path + " > " + info.text);

    EXPECT_TRUE(silent.isEmpty())
        << silent.size() << " menu entries say nothing when hovered: "
        << silent.join(", ").toStdString();
}

// The welcome page is not the editor: while it is shown the title used to
// claim "Editor - *unknown*", which reads as a bug ("what is unknown?") on
// the first screen a new user ever sees.
TEST_F(MainWindow, TheWindowTitleFollowsTheWelcomePage)
{
    QAction *welcome = action("&Welcome Screen");
    ASSERT_NE(welcome, nullptr) << "the Welcome Screen entry is gone";
    EXPECT_EQ(welcome->shortcut(), QKeySequence("Alt+Home"));

    welcome->trigger();
    QCoreApplication::processEvents();
    EXPECT_EQ(gui->windowTitle().toStdString(), "SPARTA-GUI - Welcome");

    // leaving the welcome page brings the editor title back
    QAction *fresh = action("&New Input File");
    ASSERT_NE(fresh, nullptr);
    fresh->trigger();
    QCoreApplication::processEvents();
    EXPECT_TRUE(gui->windowTitle().contains("Editor"))
        << gui->windowTitle().toStdString();
}

int main(int argc, char **argv)
{
    // offscreen, so this needs no display and no window manager
    qputenv("QT_QPA_PLATFORM", "offscreen");
    // A native file or colour picker runs its own event loop that nothing here
    // can reach into; Qt's own dialogs are ordinary widgets.
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    // main() is deliberately not linked here, so this stands in for the part of
    // it the window reads while constructing itself.
    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT  = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    // Settings of this process's own, in a directory of its own.
    //
    // Without this every case shares one scope. That is harmless while the
    // whole binary is one process, which is how the VTK build runs it -- but
    // the build without VTK discovers the cases individually and ctest runs
    // them side by side, so one case's clear() wipes the plugin path another
    // has just written and that one then sits on the missing-library dialog
    // for as long as ctest will wait.
    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_mainwindow-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
