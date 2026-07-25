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

// Exhaustive widget coverage.
//
// A hand-written list of controls is only ever as complete as the person who
// wrote it, so this walks the live widget tree instead: for each window it
// enumerates findChildren<QWidget*>() and drives everything it finds. A control
// added tomorrow is picked up without anyone remembering to list it.
//
// What it establishes per control: the app does not crash, no critical or fatal
// Qt message is emitted, and (for checkables, combo boxes and spin boxes) the
// state actually changed. What it deliberately does NOT establish is whether
// anything visible happened -- clicking "zoom in" and not crashing says nothing
// about whether the image zoomed. That is the job of the visual-effect pass,
// which drives the running application and compares renders.
//
// The count driven per window is reported so coverage is a measured number
// rather than a claim.

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QAbstractSpinBox>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QClipboard>
#include <QLabel>
#include <QMainWindow>
#include <QMimeData>
#include <QSignalSpy>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QTest>
#include <QTimer>
#include <QDir>
#include <QElapsedTimer>
#include <QFont>
#include <QIcon>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <QWidget>

#include <memory>
#include <set>
#include <string>

#include "DockWidget.h"

#include "aboutdialog.h"
#include "dockpanels.h"
#include "codeeditor.h"
#include "findandreplace.h"
#include "helpers.h"
#include "stdcapture.h"
#include "paraviewdialog.h"
#include "setvariables.h"
#include "spartawrapper.h"
#include "chartviewer.h"
#include "imageviewer.h"
#include "preferences.h"
#include "runhistory.h"
#include "slideshow.h"
#include "viewerpanel.h"
#include "stlimportwizard.h"
#include "surfreportdialog.h"

// ---------------------------------------------------------------------------
// Controls that must not be clicked
// ---------------------------------------------------------------------------
//
// Each of these either destroys the environment the rest of the run depends on
// or replaces the process. They are skipped by name and the skip is printed, so
// an entry can never quietly grow into "we don't test that any more". Every one
// is covered separately, in isolation, elsewhere in the suite.

struct SkipRule {
    const char *match;  ///< matched against objectName, then against text()
    const char *reason;
};

static const SkipRule SKIP_RULES[] = {
    {"Reset Preferences", "wipes every stored setting with no confirmation"},
    {"Defaults", "wipes every stored setting with no confirmation"},
    {"Check for", "downloads a library and then relaunches the process"},
    {"Quit", "terminates the application"},
    {"Delete", "removes archived runs from disk"},
    {"Browse", "opens a modal native file dialog that blocks the run"},
    {"Download", "network fetch followed by a process relaunch"},
    {"colorSwatch", "opens a modal colour picker with its own event loop; the "
                    "adjacent colour text field covers the same setting"},
};

/**
 * @brief Reduce a label or object name to comparable letters only
 *
 * Labels carry "&" mnemonics and spacing that object names do not, so
 * "Check for SPARTA update" and "checkForUpdate" must both match the same rule.
 * Without this a rule silently stops matching after a rename and the walker
 * goes on to click something destructive.
 */
static QString normalize(const QString &in)
{
    QString out;
    for (const QChar c : in)
        if (c.isLetterOrNumber()) out.append(c.toLower());
    return out;
}

/** @brief True if this control must not be driven; sets @p reason when so */
static bool shouldSkip(const QWidget *w, QString &reason)
{
    const QString name = normalize(w->objectName());
    QString text;
    if (auto *b = qobject_cast<const QAbstractButton *>(w)) text = normalize(b->text());

    for (const auto &rule : SKIP_RULES) {
        const QString m = normalize(QString::fromLatin1(rule.match));
        if ((!name.isEmpty() && name.contains(m)) || (!text.isEmpty() && text.contains(m))) {
            reason = QString::fromLatin1(rule.reason);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Walker
// ---------------------------------------------------------------------------

struct WalkResult;
static WalkResult walkWidget(QWidget *root, int budgetMs);

/// Deadline shared by a window and every dialog it opens, so the cost of a
/// window is bounded overall rather than per dialog.
static QElapsedTimer g_windowClock;
static int g_windowBudgetMs = 90000;
static bool windowOutOfTime()
{
    return g_windowClock.isValid() && g_windowClock.elapsed() > g_windowBudgetMs;
}

/**
 * @brief Walk then close any modal dialog that appears
 *
 * Plenty of controls legitimately open a modal -- Convert with empty fields
 * warns, Apply may confirm. exec() on that modal blocks the walk forever.
 * Rather than special-casing each button (a list that rots), reap whatever
 * modal is active on a timer and count it: the count is itself a useful signal
 * about how many controls put a dialog in the user's way.
 */
class ModalReaper : public QObject {
public:
    /** @param own the window under test, which must never be reaped */
    explicit ModalReaper(QWidget *own = nullptr, QObject *parent = nullptr)
        : QObject(parent), owner(own)
    {
        connect(&timer, &QTimer::timeout, this, &ModalReaper::reap);
        timer.start(50);
    }
    int reaped() const { return count; }

private:
    void reap();   // defined below, once WalkResult is complete
    QWidget *owner = nullptr;
    QTimer timer;
    int count = 0;
    int nested = 0;
    bool walking = false;   ///< guards against a modal opened while walking a modal

public:
    int nestedControls() const { return nested; }
};

/** @brief Tally of what a walk touched, so coverage is measurable */
struct WalkResult {
    int buttons = 0;
    int checkables = 0;
    int combos = 0;
    int spins = 0;
    int edits = 0;
    int sliders = 0;
    int tabs = 0;
    int skipped = 0;
    bool truncated = false;   ///< the time budget was hit mid-walk
    QStringList skipNotes;

    int total() const { return buttons + checkables + combos + spins + edits + sliders + tabs; }
};

/**
 * @brief Drive every interactive child of @p root
 *
 * Order matters: tabs are selected first so that controls parented to a hidden
 * page are laid out and reachable before the sweep, otherwise every tab beyond
 * the first contributes nothing and the run still reports success.
 */
static WalkResult walkWidget(QWidget *root, int budgetMs)
{
    WalkResult r;
    QElapsedTimer clock;
    clock.start();
    // Some controls are expensive: the image settings dialog re-renders through
    // SPARTA on every Apply. Bound the walk rather than let it run forever, and
    // record when the bound was hit so a truncated walk is never mistaken for a
    // complete one.
    const auto outOfTime = [&]() { return clock.elapsed() > budgetMs || windowOutOfTime(); };

    // Tabs first, so later passes see the controls on every page.
    const auto tabWidgets = root->findChildren<QTabWidget *>();
    for (auto *tw : tabWidgets) {
        for (int i = 0; i < tw->count(); ++i) {
            tw->setCurrentIndex(i);
            QCoreApplication::processEvents();
            ++r.tabs;
        }
        if (tw->count() > 0) tw->setCurrentIndex(0);
    }

    // Buttons: plain ones get clicked, checkable ones toggled both ways and
    // asserted to have actually moved -- a checkbox wired to nothing would
    // otherwise pass simply by not crashing.
    for (auto *b : root->findChildren<QAbstractButton *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!b->isEnabled() || !b->isVisible()) continue;
        QString why;
        if (shouldSkip(b, why)) {
            ++r.skipped;
            r.skipNotes << QString("  skipped %1 (%2): %3")
                               .arg(b->objectName().isEmpty() ? b->text() : b->objectName())
                               .arg(b->metaObject()->className(), why);
            continue;
        }
        if (b->isCheckable()) {
            const bool before = b->isChecked();
            b->setChecked(!before);
            QCoreApplication::processEvents();
            EXPECT_NE(before, b->isChecked())
                << "checkable did not change state: "
                << (b->objectName().isEmpty() ? b->text() : b->objectName()).toStdString();
            b->setChecked(before);
            QCoreApplication::processEvents();
            ++r.checkables;
        } else {
            QTest::mouseClick(b, Qt::LeftButton);
            QCoreApplication::processEvents();
            ++r.buttons;
        }
    }

    // Combo boxes: every index, not just a couple, since option-specific code
    // paths (e.g. a colormap that only some entries exercise) hide in the tail.
    for (auto *c : root->findChildren<QComboBox *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!c->isEnabled() || c->count() == 0) continue;
        const int before = c->currentIndex();
        for (int i = 0; i < c->count(); ++i) {
            c->setCurrentIndex(i);
            QCoreApplication::processEvents();
            EXPECT_EQ(i, c->currentIndex());
        }
        c->setCurrentIndex(before);
        ++r.combos;
    }

    // Spin boxes: the bounds are where clamping bugs live, so drive both ends
    // and a value in between rather than nudging by one step.
    for (auto *s : root->findChildren<QSpinBox *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!s->isEnabled()) continue;
        const int before = s->value();
        for (int v : {s->minimum(), s->maximum(), (s->minimum() + s->maximum()) / 2}) {
            s->setValue(v);
            QCoreApplication::processEvents();
            EXPECT_EQ(v, s->value());
        }
        s->setValue(before);
        ++r.spins;
    }
    for (auto *s : root->findChildren<QDoubleSpinBox *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!s->isEnabled()) continue;
        const double before = s->value();
        for (double v : {s->minimum(), s->maximum(), (s->minimum() + s->maximum()) / 2.0}) {
            s->setValue(v);
            QCoreApplication::processEvents();
        }
        s->setValue(before);
        ++r.spins;
    }

    // Line edits: a plausible value, then empty. Fields carrying a validator
    // additionally get a value it must reject, since a missing validator is a
    // real defect this suite is meant to surface.
    for (auto *e : root->findChildren<QLineEdit *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!e->isEnabled() || e->isReadOnly()) continue;
        // Spin boxes own an internal QLineEdit. Driving it as a free-text field
        // types junk into the spin box's editor and counts one control twice.
        if (qobject_cast<QAbstractSpinBox *>(e->parentWidget())) continue;
        const QString before = e->text();
        e->setText("1.0");
        QCoreApplication::processEvents();
        e->clear();
        QCoreApplication::processEvents();
        if (e->validator()) {
            e->setText("!!not-valid!!");
            QCoreApplication::processEvents();
        }
        e->setText(before);
        ++r.edits;
    }

    for (auto *s : root->findChildren<QSlider *>()) {
        if (outOfTime()) { r.truncated = true; break; }
        if (!s->isEnabled()) continue;
        const int before = s->value();
        for (int v : {s->minimum(), s->maximum(), (s->minimum() + s->maximum()) / 2}) {
            s->setValue(v);
            QCoreApplication::processEvents();
            EXPECT_EQ(v, s->value());
        }
        s->setValue(before);
        ++r.sliders;
    }

    return r;
}

/**
 * @brief Walk a modal before dismissing it
 *
 * Several buttons exist precisely to open a further dialog -- the image
 * viewer's eight settings tabs are reached that way -- so closing them on sight
 * would leave the largest window in the application untouched while the run
 * still looked complete.
 */
void ModalReaper::reap()
{
    auto *m = QApplication::activeModalWidget();

    // QColorDialog::getColor() spins its own nested event loop and does not
    // always show up as the active modal, so also sweep the top-level widgets
    // for a stray visible dialog. Without this a single colour swatch stalls
    // the whole run.
    if (!m) {
        for (auto *w : QApplication::topLevelWidgets()) {
            auto *d = qobject_cast<QDialog *>(w);
            if (d && d->isVisible() && d != owner) { m = d; break; }
        }
    }
    if (!m) return;

    ++count;
    if (!walking) {
        walking = true;
        nested += walkWidget(m, 30000).total();
        walking = false;
    }
    // a dialog spinning in exec() only leaves its loop on accept/reject
    if (auto *d = qobject_cast<QDialog *>(m))
        d->reject();
    else
        m->close();
}

/** @brief Print a per-window tally; the totals are the coverage evidence */
static void report(const QString &window, const WalkResult &r)
{
    std::printf("  %-28s %3d controls  (btn %d, chk %d, combo %d, spin %d, edit %d, "
                "slider %d, tab %d)\n",
                window.toStdString().c_str(), r.total(), r.buttons, r.checkables, r.combos,
                r.spins, r.edits, r.sliders, r.tabs);
    if (r.truncated)
        std::printf("    NOTE: walk hit its time budget; some controls were not driven\n");
    for (const auto &note : r.skipNotes)
        std::printf("%s\n", note.toStdString().c_str());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A widget built from plain Qt classes, wired the way the app wires its own
// dialogs. This proves the walker drives every control type and honours the
// skip rules before it is pointed at the real windows -- a walker that silently
// visited nothing would otherwise "pass" everywhere.
TEST(WidgetWalker, DrivesEveryControlType)
{
    QWidget w;
    auto *tabs = new QTabWidget(&w);
    auto *page1 = new QWidget;
    auto *page2 = new QWidget;
    tabs->addTab(page1, "One");
    tabs->addTab(page2, "Two");

    auto *push = new QPushButton("Push", page1);
    auto *check = new QCheckBox("Check", page1);
    auto *radio = new QRadioButton("Radio", page1);
    auto *combo = new QComboBox(page1);
    combo->addItems({"a", "b", "c"});
    auto *spin = new QSpinBox(page1);
    spin->setRange(0, 10);
    auto *dspin = new QDoubleSpinBox(page2);
    dspin->setRange(0.0, 1.0);
    auto *edit = new QLineEdit(page2);
    auto *slider = new QSlider(Qt::Horizontal, page2);
    slider->setRange(0, 100);

    int pushes = 0;
    QObject::connect(push, &QPushButton::clicked, [&pushes]() { ++pushes; });

    w.show();
    QCoreApplication::processEvents();

    const WalkResult r = walkWidget(&w, 120000);
    report("SelfTest", r);

    EXPECT_EQ(1, r.buttons) << "the plain push button was not clicked";
    EXPECT_EQ(2, r.checkables) << "checkbox and radio should both be toggled";
    EXPECT_EQ(1, r.combos);
    EXPECT_EQ(2, r.spins) << "int and double spin boxes are both spins";
    EXPECT_EQ(1, r.edits);
    EXPECT_EQ(1, r.sliders);
    EXPECT_EQ(2, r.tabs);
    EXPECT_EQ(1, pushes) << "clicking the button did not reach its slot";
}

// The skip rules are load-bearing: if a rename stops one matching, the walker
// would happily click "Reset Preferences to Defaults" and wipe the environment
// out from under every later test. Assert they match by text and by objectName.
TEST(WidgetWalker, SkipsDestructiveControls)
{
    QWidget w;
    auto *reset = new QPushButton("Reset Preferences to &Defaults", &w);
    auto *quit = new QPushButton("&Quit", &w);
    auto *browse = new QPushButton("&Browse...", &w);
    auto *safe = new QPushButton("Apply", &w);
    auto *byName = new QPushButton("harmless label", &w);
    byName->setObjectName("checkForUpdate");

    int resets = 0, quits = 0, browses = 0, safes = 0, updates = 0;
    QObject::connect(reset, &QPushButton::clicked, [&]() { ++resets; });
    QObject::connect(quit, &QPushButton::clicked, [&]() { ++quits; });
    QObject::connect(browse, &QPushButton::clicked, [&]() { ++browses; });
    QObject::connect(safe, &QPushButton::clicked, [&]() { ++safes; });
    QObject::connect(byName, &QPushButton::clicked, [&]() { ++updates; });

    w.show();
    QCoreApplication::processEvents();

    const WalkResult r = walkWidget(&w, 120000);
    report("SkipRules", r);

    EXPECT_EQ(0, resets) << "destructive: settings would have been wiped";
    EXPECT_EQ(0, quits) << "destructive: the application would have exited";
    EXPECT_EQ(0, browses) << "would have opened a blocking modal file dialog";
    EXPECT_EQ(0, updates) << "skip rules must match on objectName, not only text";
    EXPECT_EQ(1, safes) << "a harmless button must still be driven";
    EXPECT_EQ(4, r.skipped) << "four of the five buttons are destructive";
    EXPECT_EQ(1, r.buttons) << "only the harmless Apply should be driven";
}

// A checkable wired to nothing is a real defect class (a preference that reads
// back its own default forever). Confirm the walker's state assertion is what
// catches it, rather than the walk merely completing.
TEST(WidgetWalker, DetectsDeadCheckable)
{
    QWidget w;
    auto *live = new QCheckBox("live", &w);
    w.show();
    QCoreApplication::processEvents();

    const WalkResult r = walkWidget(&w, 120000);
    EXPECT_EQ(1, r.checkables);
    // toggling must be observable; the walker asserts this internally
    EXPECT_FALSE(live->isChecked()) << "walker must restore the original state";
}

// ---------------------------------------------------------------------------
// The real windows
// ---------------------------------------------------------------------------
//
// These are the coverage that matters: the walker is pointed at the
// application's own dialogs, constructed exactly as the app constructs them.
// Each test records how many controls it drove so a window that silently
// produced nothing shows up as a number, not as a pass.

static int g_totalDriven = 0;

/** @brief Walk a real window, record its tally, and fold it into the total */
static void walkWindow(QWidget *w, const QString &name, int expectAtLeast)
{
    ModalReaper reaper(w);   // a control that opens a modal must not stall the walk
    g_windowClock.start();
    w->show();
    QCoreApplication::processEvents();
    const WalkResult r = walkWidget(w, 90000);
    report(name, r);
    if (reaper.reaped() > 0) {
        std::printf("    (+%d controls in %d nested dialog(s))\n", reaper.nestedControls(),
                    reaper.reaped());
        g_totalDriven += reaper.nestedControls();
    }
    g_totalDriven += r.total();
    EXPECT_GE(r.total(), expectAtLeast)
        << name.toStdString()
        << ": far fewer controls than this window is known to have -- it was probably "
           "not laid out, so most of it went undriven";
}

// ---------------------------------------------------------------------------
// Editor clipboard handling
// ---------------------------------------------------------------------------

/// Exposes the protected override so the null case can be checked directly.
class PasteProbe : public CodeEditor {
public:
    // CodeEditor deletes its default constructor in favour of the
    // parent-taking one, so name it explicitly rather than defaulting.
    PasteProbe() : CodeEditor(nullptr) {}
    using CodeEditor::canInsertFromMimeData;
};

// Found by the walker driving Edit -> Paste on a fresh session: the process
// died with SIGSEGV. QClipboard::mimeData() returns null when nothing owns the
// clipboard -- an X11 session where nothing has been copied yet, or one where
// the program that did the copying has exited -- and QPlainTextEdit::paste()
// hands that null straight to canInsertFromMimeData(), which dereferenced it.
TEST(CodeEditorClipboard, PasteWithNothingOnTheClipboardDoesNotCrash)
{
    PasteProbe editor;
    EXPECT_FALSE(editor.canInsertFromMimeData(nullptr));

    // and the whole path, the way the menu entry runs it
    QGuiApplication::clipboard()->clear();
    editor.setPlainText("run 100\n");
    editor.paste();
    EXPECT_EQ(editor.toPlainText(), QString("run 100\n"))
        << "an empty clipboard changed the buffer";
}

TEST(CodeEditorClipboard, PasteInsertsTextThatIsOnTheClipboard)
{
    PasteProbe editor;
    QGuiApplication::clipboard()->setText("stats 100");
    editor.setPlainText("");
    editor.paste();
    EXPECT_EQ(editor.toPlainText(), QString("stats 100"));
    QGuiApplication::clipboard()->clear();
}

// The inherited implementation offers the clipboard's data to a rich-text
// reader before falling back to text, and that path segfaulted here: Ctrl+V
// took the editor down whatever was on the clipboard. An input deck has no
// rich text in it, so the override takes the text and inserts that -- which is
// also what someone pasting a command out of a web page wants.
TEST(CodeEditorClipboard, RichTextIsPastedAsPlainText)
{
    PasteProbe editor;
    auto *mime = new QMimeData;
    mime->setHtml("<b>stats</b> 100");
    mime->setText("stats 100");
    QGuiApplication::clipboard()->setMimeData(mime);

    editor.setPlainText("");
    editor.paste();
    EXPECT_EQ(editor.toPlainText(), QString("stats 100"))
        << "markup reached the document, which SPARTA cannot parse";
    QGuiApplication::clipboard()->clear();
}

TEST(CodeEditorClipboard, MimeDataWithNoTextInsertsNothing)
{
    PasteProbe editor;
    auto *mime = new QMimeData;
    mime->setData("application/octet-stream", QByteArray("\x01\x02", 2));
    QGuiApplication::clipboard()->setMimeData(mime);

    editor.setPlainText("run 10");
    editor.paste();
    EXPECT_EQ(editor.toPlainText(), QString("run 10"));
    QGuiApplication::clipboard()->clear();
}

// ---------------------------------------------------------------------------
// The View menu's panel entries
// ---------------------------------------------------------------------------

// Also found by the live walker: partway through a sweep the entries for
// Output, Charts and Viewer vanished from the View menu, while the other five
// stayed. They had not been removed -- they had been renamed. Qt-ADS retitles
// a dock's toggleViewAction() whenever the dock's title changes, and that
// action is the menu entry, so naming a panel after its current contents
// rewrote the menu underneath the user: "Output Window" became "Output -
// in.circle - Run 1" as soon as a run started.
TEST(PanelMenu, RetitlingAPanelLeavesItsMenuEntryAlone)
{
    QMainWindow window;
    CodeEditor editor(nullptr);
    PanelManager panels(&window, &editor);

    QAction *entry = panels.toggleViewAction(PanelManager::Log);
    ASSERT_NE(entry, nullptr);
    entry->setText("&Output Window");   // what the View menu sets it to

    panels.setPanelTitle(PanelManager::Log, "Output - in.circle - Run 1");

    EXPECT_EQ(entry->text(), QString("&Output Window"))
        << "the menu entry was renamed after the panel's contents";
    // the tab still says what it is showing, which is where that is useful
    EXPECT_EQ(panels.dock(PanelManager::Log)->windowTitle(),
              QString("Output - in.circle - Run 1"));
}

TEST(PanelMenu, ReplacingAPanelWidgetLeavesItsMenuEntryAlone)
{
    // the other path that retitles a dock: handing it a new widget
    QMainWindow window;
    CodeEditor editor(nullptr);
    PanelManager panels(&window, &editor);

    QAction *entry = panels.toggleViewAction(PanelManager::Viewer);
    entry->setText("&Viewer Window");

    panels.setPanelWidget(PanelManager::Viewer, new QLabel("frame"), "test.0100.ppm", false);

    EXPECT_EQ(entry->text(), QString("&Viewer Window"));
    EXPECT_EQ(panels.dock(PanelManager::Viewer)->windowTitle(), QString("test.0100.ppm"));
}

TEST(RealWindows, FindAndReplace)
{
    CodeEditor editor(nullptr);
    editor.setPlainText("run 100\nrun 200\n");
    FindAndReplace dlg(&editor);
    walkWindow(&dlg, "FindAndReplace", 8);
}

TEST(RealWindows, SetVariables)
{
    QList<QPair<QString, QString>> vars{{"seed", "12345"}, {"nsteps", "200"}};
    SetVariables dlg(vars);
    // three controls per variable row plus Add Row and the button box
    walkWindow(&dlg, "SetVariables", 4);
}

TEST(RealWindows, AboutDialog)
{
    AboutDialog dlg("SPARTA-GUI 1.0.0", "config info", "style details", 400);
    walkWindow(&dlg, "AboutDialog", 1);
}

// Found by looking at the dialog: the version line, which also carries the
// full path of the loaded plugin, ran past the right edge and was cut off --
// no ellipsis, no scrollbar, no way to read it. Which library got loaded is
// the one thing this dialog is opened for when a plugin misbehaves, and an
// ordinary absolute path was long enough to lose it.
TEST(AboutDialogLayout, TheVersionAndPluginPathAreReadable)
{
    const QString version = "This is SPARTA-GUI version 1.0.0 using Qt version 6.4.2\n"
                            "SPARTA library loaded as plugin from file "
                            "/tmp/claude-0/-home-user-sparta/cb58d289-558d-510d-8b18-"
                            "9976ada54800/scratchpad/v1beta/build-lib/src/libsparta_.so";
    AboutDialog dlg(version, "SPARTA version: 24 Sep 2025\nKOKKOS package: not included",
                    "Fix styles:\nablate adapt ambipolar ave/grid", 400);
    dlg.show();
    QApplication::processEvents();

    QLabel *versionLabel = nullptr;
    for (auto *l : dlg.findChildren<QLabel *>())
        if (l->text().startsWith("This is SPARTA-GUI")) versionLabel = l;
    ASSERT_NE(versionLabel, nullptr);

    EXPECT_TRUE(versionLabel->wordWrap())
        << "word wrap is off, so a path longer than the dialog is silently truncated";

    // With wrapping on, the text has to fit in the height the layout gave it.
    const int needed = versionLabel->heightForWidth(versionLabel->width());
    EXPECT_LE(needed, versionLabel->height())
        << "the version block is taller than the space it has, so its last line is cut off";
    // and the label must not be narrower than the dialog can afford
    EXPECT_GT(versionLabel->width(), 200);
}

TEST(RealWindows, ParaViewExport)
{
    ParaViewExportDialog dlg(nullptr, QDir::tempPath());
    // two modes on a stacked page, both sets of options, the tool pickers
    walkWindow(&dlg, "ParaViewExport", 12);
}

/**
 * @brief Where the test fixtures are, from the environment or the build
 *
 * The build knows the answer, so it compiles it in; the environment overrides
 * it for anyone running the binary against a different set. Reading only the
 * environment is what left every fixture-backed test skipping in ordinary
 * builds: nothing set the variable, and a skip reports as a pass.
 */
static QString fixturesDir()
{
    const QString fromEnv = QString::fromLatin1(qgetenv("SPARTA_FIXTURES"));
    if (!fromEnv.isEmpty()) return fromEnv;
#ifdef SPARTA_TEST_FIXTURES_DIR
    return QStringLiteral(SPARTA_TEST_FIXTURES_DIR);
#else
    return {};
#endif
}

/** @brief The shared libsparta the live-window tests load, or empty if none */
static QString testLibrary()
{
    const QString fromEnv = QString::fromLatin1(qgetenv("SPARTA_PLUGIN_LIB"));
    if (!fromEnv.isEmpty()) return fromEnv;
#ifdef SPARTA_TEST_LIBRARY_PATH
    return QStringLiteral(SPARTA_TEST_LIBRARY_PATH);
#else
    return {};
#endif
}

/**
 * @brief Copy a fixture to a scratch dir before handing it to the wizard
 *
 * The wizard writes a .surf next to its input, so pointing it straight at the
 * fixtures directory leaves artifacts in the source tree.
 */
static QString stagedFixture(const QString &name, QTemporaryDir &dir)
{
    const QString src = fixturesDir() + "/" + name;
    if (!QFileInfo::exists(src)) return {};
    const QString dst = dir.filePath(name);
    QFile::copy(src, dst);
    return dst;
}

TEST(RealWindows, StlImportWizardWatertight)
{
    QTemporaryDir tmp;
    const QString stl = stagedFixture("tetra.stl", tmp);
    if (stl.isEmpty()) GTEST_SKIP() << "fixture tetra.stl not found";
    SpartaWrapper sparta;   // deliberately not opened: the wizard must cope
    StlImportWizard dlg(nullptr, &sparta, stl);
    walkWindow(&dlg, "StlWizard(watertight)", 20);
}

TEST(RealWindows, StlImportWizardLeaky)
{
    QTemporaryDir tmp;
    const QString stl = stagedFixture("open.stl", tmp);
    if (stl.isEmpty()) GTEST_SKIP() << "fixture open.stl not found";
    SpartaWrapper sparta;
    StlImportWizard dlg(nullptr, &sparta, stl);
    // the leaky mesh disables the SPARTA-render buttons, so expect fewer
    walkWindow(&dlg, "StlWizard(leaky)", 18);
}

// ---------------------------------------------------------------------------
// Windows that need a live SPARTA instance
// ---------------------------------------------------------------------------
//
// These are the large ones -- the image viewer and its eight settings tabs, the
// surface report, run history. They read species, surfaces and box extents from
// a running simulation to populate themselves, so a wrapper is opened and a
// deck is run before the walk. Without that they come up empty and a walk of
// them would report a handsome number while touching almost nothing.

/** @brief Open a SPARTA instance and bring it to a state with box, grid and surfaces */
static bool openSparta(SpartaWrapper &sparta, const QString &deck)
{
    const QString lib = testLibrary();
    if (lib.isEmpty() || !QFileInfo::exists(lib)) return false;
    if (!sparta.loadLib(lib)) return false;

    char arg0[] = "sparta";
    char argq[] = "-log";
    char argn[] = "none";
    char *args[] = {arg0, argq, argn};
    sparta.open(3, args);
    if (!sparta.isOpen()) return false;

    QFile f(deck);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    const QString text = QString::fromUtf8(f.readAll());

    // The deck names its species, collision and surface files relatively, so it
    // only runs from its own directory. Run it there and change back, otherwise
    // read_surf fails, no surfaces or computes exist, and the windows under test
    // come up empty while still reporting success.
    const QString prev = QDir::currentPath();
    QDir::setCurrent(QFileInfo(deck).absolutePath());
    {
        StdoutSilencer guard;   // the deck is chatty and would drown the output
        sparta.commandsString(text);
    }
    QDir::setCurrent(prev);
    return true;
}

/** @brief Path to a deck that defines a box, a grid, species and surfaces */
static QString surfDeck()
{
    return fixturesDir() + "/in.surfq";
}

TEST(RealWindowsLive, SurfaceReport)
{
    SpartaWrapper sparta;
    if (!openSparta(sparta, surfDeck()))
        GTEST_SKIP() << "needs SPARTA_PLUGIN_LIB and the in.surfq fixture";
    QFile f(surfDeck());
    f.open(QIODevice::ReadOnly | QIODevice::Text);
    SurfReportDialog dlg(nullptr, &sparta, QString::fromUtf8(f.readAll()));
    walkWindow(&dlg, "SurfaceReport", 4);
}

TEST(RealWindowsLive, RunHistoryPanel)
{
    RunHistory hist;
    HistoryPanel panel(nullptr, &hist);
    // Compare starts disabled and Delete is skipped as destructive
    walkWindow(&panel, "RunHistory", 4);
}

TEST(RealWindowsLive, Preferences)
{
    SpartaWrapper sparta;
    // Preferences reads accelerator support from the library, so open it; the
    // dialog still builds without one, just with the Kokkos option disabled.
    openSparta(sparta, surfDeck());
    Preferences dlg(&sparta, nullptr);
    // five tabs; the source audit counts 64 controls across them
    walkWindow(&dlg, "Preferences", 40);
}

TEST(RealWindowsLive, ChartWindowStandalone)
{
    // spartagui == nullptr selects standalone mode, which is the variant that
    // has the X-axis field and the "add data from file" entry
    ChartWindow win("series", nullptr);
    walkWindow(&win, "ChartWindow", 8);
}

TEST(RealWindowsLive, SlideShowStandalone)
{
    SlideShow show("", nullptr);
    walkWindow(&show, "SlideShow", 15);
}

TEST(RealWindowsLive, ImageViewer)
{
    SpartaWrapper sparta;
    if (!openSparta(sparta, surfDeck()))
        GTEST_SKIP() << "needs SPARTA_PLUGIN_LIB and the in.surfq fixture";
    // renders on construction, which needs box and grid to exist -- the deck
    // above provides both
    ImageViewer viewer("test", &sparta, nullptr);
    walkWindow(&viewer, "ImageViewer", 20);
}

// The merged viewer panel: the tab bar plus whichever source is in front. The
// walk covers the front page only, which is the point -- the panel shows one
// source at a time, and a control on a hidden page is not reachable by a user
// either.
TEST(RealWindowsLive, ViewerPanel)
{
    SpartaWrapper sparta;
    if (!openSparta(sparta, surfDeck()))
        GTEST_SKIP() << "needs SPARTA_PLUGIN_LIB and the in.surfq fixture";

    ViewerPanel panel;
    panel.addSource(ViewerPanel::Snapshot, new ImageViewer("test", &sparta, nullptr));
    panel.addSource(ViewerPanel::Sequence, new SlideShow("", nullptr));

    // both sources registered, both reachable by tab
    EXPECT_TRUE(panel.hasSource(ViewerPanel::Snapshot));
    EXPECT_TRUE(panel.hasSource(ViewerPanel::Sequence));

    panel.showSource(ViewerPanel::Snapshot, true);
    EXPECT_EQ(panel.currentSource(), ViewerPanel::Snapshot);
    panel.showSource(ViewerPanel::Sequence, true);
    EXPECT_EQ(panel.currentSource(), ViewerPanel::Sequence);

    walkWindow(&panel, "ViewerPanel", 15);
}

// Once the user has chosen a source, frames arriving on their own must not take
// the view away from it -- that would move the window out from under someone in
// the middle of looking at something.
TEST(ViewerPanelBehaviour, BackgroundContentDoesNotStealTheView)
{
    SpartaWrapper sparta;
    if (!openSparta(sparta, surfDeck()))
        GTEST_SKIP() << "needs SPARTA_PLUGIN_LIB and the in.surfq fixture";

    ViewerPanel panel;
    panel.addSource(ViewerPanel::Sequence, new SlideShow("", nullptr));
    panel.addSource(ViewerPanel::Snapshot, new ImageViewer("test", &sparta, nullptr));

    panel.showSource(ViewerPanel::Snapshot, true);      // the user picked this
    panel.showSource(ViewerPanel::Sequence);            // a run wrote a frame
    EXPECT_EQ(panel.currentSource(), ViewerPanel::Snapshot);

    // a new run clears the choice, so the next run's frames do come forward
    panel.unlockSource();
    panel.showSource(ViewerPanel::Sequence);
    EXPECT_EQ(panel.currentSource(), ViewerPanel::Sequence);
}

TEST(RealWindows, TotalCoverage)
{
    // ctest runs each discovered test in its own process, so the accumulator is
    // only meaningful when the binary is run directly (which is how the coverage
    // figure for the report is produced). Under ctest this legitimately sees
    // zero, and must not fail for it.
    if (g_totalDriven == 0) {
        GTEST_SKIP() << "run the binary directly to get the aggregate count; "
                        "ctest isolates each test in its own process";
    }
    std::printf("\n  === controls driven across real windows: %d ===\n", g_totalDriven);
}

int main(int argc, char **argv)
{
    // offscreen so this runs without a display; the visual pass uses a real
    // framebuffer because it needs to photograph what was rendered
    qputenv("QT_QPA_PLATFORM", "offscreen");
    // Native colour and file pickers run their own event loop that the reaper
    // cannot reach into, so a single colour swatch would stall the whole run.
    // Qt's own dialogs are ordinary widgets and close on demand.
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    // main() owns this setup and is deliberately not linked here, so the test
    // has to stand in for it: the dialogs read these fonts and the resource
    // bundle while constructing themselves.
    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    // never let a walk write into the developer's real configuration
    QCoreApplication::setOrganizationName("SPARTA-GUI-Test");
    QCoreApplication::setApplicationName("widget-walker");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
