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

// The run path: what the main window does while and after a simulation runs.
//
// doRun(), archiveFinishedRun(), continueRestart() and renderVtkSnapshot() were
// the last large block of spartagui.cpp with no coverage, and the reason given
// was that they need a simulation actually running.  Three of the four do -- and
// a simulation is something this suite can have: the same shared libsparta the
// window loads anyway, driven through the window's own Run action on a deck
// small enough to finish in a moment.  The fourth, continueRestart(), turns out
// to need no simulator at all; it lists files and writes two lines into the
// editor.
//
// A run is asynchronous (SpartaRunner is a thread), so every case here waits on
// the runFinished signal rather than assuming the run is over when the call
// returns.

#include "spartagui.h"

#include "chartviewer.h"
#include "codeeditor.h"
#include "constants.h"
#include "logwindow.h"
#include "runhistory.h"

#include <gtest/gtest.h>

#include <QAction>
#include <QAbstractButton>
#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QIcon>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QSettings>
#include <QSignalSpy>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>

#include <memory>

namespace {

// A deck that creates a box, a species and a run, and finishes immediately.
// Everything the run path touches -- thermo output, the log window, the chart
// columns, the archive record -- comes from this.
const char *const kDeck = "seed            12345\n"
                          "dimension       2\n"
                          "global          gridcut 0.0 comm/sort yes\n"
                          "boundary        o r p\n"
                          "create_box      0 10 0 10 -0.5 0.5\n"
                          "create_grid     4 4 1\n"
                          "balance_grid    rcb cell\n"
                          "species         ar.species Ar\n"
                          "mixture         air Ar vstream 0.0 0.0 0.0\n"
                          "global          nrho 1.0 fnum 1.0\n"
                          "create_particles air n 100\n"
                          "collide         vss air ar.vss\n"
                          "stats           1\n"
                          "run             5\n";

// The two data files the deck reads, copied verbatim from SPARTA's own data/
// directory. Written here rather than located at run time so the test does not
// depend on where the examples happen to be installed -- and taken from the
// real files rather than invented, because the species parser rejects a line
// with the wrong number of fields and the run then fails for the wrong reason.
const char *const kSpecies = "# Species data: ID, molwt, molmass, rotdof, rotrel,\n"
                             "# vibdof, vibrel, vibtemp, species wt, charge\n"
                             "Ar  40.00    6.63E-26  0    .0   0   .0    0.0    1.0      0.0\n";
const char *const kVss     = "# VSS collision parameters: diameter, omega, tref, alpha\n"
                             "Ar   4.11e-10 0.81  273.15  1.4\n";

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

// Dismisses anything modal, remembering what it said.  Nothing here should
// raise one on the happy path; when something does, that is the finding.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 30000) : left(budgetMs)
    {
        timer.setInterval(10);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }

    QStringList seen;
    int count = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &s : seen)
            if (s.contains(needle)) return true;
        return false;
    }

    /// answer the next QMessageBox with this button instead of accepting
    QMessageBox::StandardButton answer = QMessageBox::NoButton;

private:
    void poll()
    {
        if ((left -= 10) < 0) {
            timer.stop();
            return;
        }
        auto *m = QApplication::activeModalWidget();
        if (!m) return;
        ++count;
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            seen << box->text() + "\n" + box->informativeText();
            if (answer != QMessageBox::NoButton) {
                if (auto *b = box->button(answer)) {
                    b->click();
                    return;
                }
            }
            box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            static_cast<QDialog *>(fd)->reject();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m))
            d->reject();
        else
            m->close();
    }

    QTimer timer;
    int left;
};

// archiveFinishedRun() is protected; a subclass is how a test reaches it
// without widening the interface for everyone else.
class TestableGui : public SpartaGui {
public:
    using SpartaGui::SpartaGui;
    using SpartaGui::openFile;
    using SpartaGui::writeFile;
};

class Run : public ::testing::Test {
protected:
    void SetUp() override
    {
        REQUIRE_LIBRARY();
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.sync();

        // the deck reads these by relative path, so the run has to happen here
        writeFile("ar.species", kSpecies);
        writeFile("ar.vss", kVss);
        // The run archive lives on disk under AppDataLocation and is reloaded by
        // every new window, so without this each case counts the records the
        // previous ones left behind.
        QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/history")
            .removeRecursively();

        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());

        bool modalSeen = false;
        QTimer reaper;
        QObject::connect(&reaper, &QTimer::timeout, [&modalSeen]() {
            if (QWidget *m = QApplication::activeModalWidget()) {
                modalSeen = true;
                m->close();
            }
        });
        reaper.start(50);
        gui = new TestableGui(nullptr, QString(), 800, 600);
        reaper.stop();
        ASSERT_FALSE(modalSeen) << "the main window put up a modal while being constructed";
    }

    void TearDown() override
    {
        if (gui) {
            QMetaObject::invokeMethod(gui, "stopRun");
            waitFor([this] { return !running(); }, 10000);
        }
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    QString writeFile(const QString &name, const QByteArray &bytes) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
        return p;
    }

    CodeEditor *editor() const { return gui->findChild<CodeEditor *>(); }
    void setBuffer(const QString &t) const { editor()->setPlainText(t); }
    bool running() const { return gui->findChild<QProgressBar *>() && spy && spy->isEmpty(); }

    /// spin the event loop until @p done or the budget runs out
    template <class F> static bool waitFor(F done, int budgetMs)
    {
        QElapsedTimer clock;
        clock.start();
        while (!done() && clock.elapsed() < budgetMs)
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        return done();
    }

    /// run the editor buffer and wait for it to finish; returns the success flag
    bool runBuffer(int budgetMs = 30000)
    {
        QSignalSpy finished(gui, &SpartaGui::runFinished);
        gui->runBuffer();
        const bool done = waitFor([&finished] { return !finished.isEmpty(); }, budgetMs);
        EXPECT_TRUE(done) << "the run did not finish within " << budgetMs << " ms";
        if (!done || finished.isEmpty()) return false;
        return finished.first().at(0).toBool();
    }

    static void collect(QMenu *menu, QList<QAction *> &out)
    {
        for (auto *a : menu->actions()) {
            out.append(a);
            if (a->menu()) collect(a->menu(), out);
        }
    }

    QList<QAction *> allActions() const
    {
        QList<QAction *> out;
        if (auto *bar = gui->findChild<QMenuBar *>())
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

    QTemporaryDir dir;
    QString startDir;
    TestableGui *gui   = nullptr;
    QSignalSpy *spy    = nullptr;
};

} // namespace

// ---------------------------------------------------------------- a run happens

TEST_F(Run, RunningTheBufferFinishesAndSaysItSucceeded)
{
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    EXPECT_TRUE(runBuffer()) << "the run reported failure: " << modals.seen.join(" | ").toStdString();
    EXPECT_EQ(modals.count, 0) << "the run put up a dialog: " << modals.seen.join(" | ").toStdString();
}

TEST_F(Run, TheRunPutsItsOutputInTheLogWindow)
{
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());

    auto *log = gui->findChild<LogWindow *>();
    ASSERT_NE(log, nullptr) << "no log window was created for the run";
    const QString text = log->toPlainText();
    // the "SPARTA (date)" banner is printed before the capture starts, so what
    // reaches the log window is the deck's own output from create_box onwards
    EXPECT_TRUE(text.contains("Created orthogonal box")) << text.left(400).toStdString();
    EXPECT_TRUE(text.contains("Step")) << "the thermo header never reached the log";
    EXPECT_TRUE(text.contains("CPU time")) << "the run summary never reached the log";
}

TEST_F(Run, TheRunFillsTheChartWithItsThermoColumns)
{
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());

    auto *chart = gui->findChild<ChartWindow *>();
    ASSERT_NE(chart, nullptr) << "no chart window was created for the run";
    EXPECT_GT(chart->numCharts(), 0) << "the run produced no plotted thermo columns";
    EXPECT_GE(chart->getStep(), 0) << "the chart holds no steps";
}

TEST_F(Run, TheRunCounterIsReadableFromTheDeckAndAdvances)
{
    // gui_run is an index variable SPARTA-GUI defines for every run so a deck
    // can name its output after the run it came from; it must not repeat
    Modals modals;
    setBuffer("print \"gui_run is ${gui_run}\"\n");

    ASSERT_TRUE(runBuffer());
    auto *log = gui->findChild<LogWindow *>();
    ASSERT_NE(log, nullptr);
    EXPECT_TRUE(log->toPlainText().contains("gui_run is 1"))
        << log->toPlainText().toStdString();

    // re-fetch: each run builds a new LogWindow, so the pointer from the first
    // run dangles the moment the second one starts
    ASSERT_TRUE(runBuffer());
    log = gui->findChild<LogWindow *>();
    ASSERT_NE(log, nullptr);
    const QString second = log->toPlainText();
    EXPECT_TRUE(second.contains("gui_run is 2"))
        << "the run counter did not advance between runs: " << second.toStdString();
    EXPECT_FALSE(second.contains("gui_run is 1"))
        << "each run gets a fresh log window, so the previous run's output should "
           "not still be there: " << second.toStdString();
}

TEST_F(Run, ADeckThatCannotRunReportsFailureRatherThanClaimingSuccess)
{
    Modals modals;
    setBuffer("this is not a SPARTA command\n");
    const bool ok = runBuffer();
    EXPECT_FALSE(ok) << "an invalid deck was reported as a successful run";
}

TEST_F(Run, TheProgressAndStatusIndicatorsComeUpForARun)
{
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());
    auto *progress = gui->findChild<QProgressBar *>();
    ASSERT_NE(progress, nullptr) << "the window has no progress bar";
}

// ---------------------------------------------------------------- the run guards

TEST_F(Run, ADeckContainingQuitAsksBeforeRunning)
{
    // SPARTA's quit command calls exit(), which would take the whole GUI with
    // it -- so this prompt is the only thing between a user and losing their
    // editor buffer.  Declining must leave the deck unrun, and doRun() only
    // builds a log window once it is past this guard, so the absence of one is
    // what "unrun" looks like from outside.
    Modals modals;
    modals.answer = QMessageBox::No;
    setBuffer(QString::fromLatin1(kDeck) + "quit\n");

    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    waitFor([&modals] { return modals.count > 0; }, 5000);

    EXPECT_TRUE(modals.said("terminate not only the SPARTA run"))
        << "the warning did not say why quit is dangerous: "
        << modals.seen.join(" | ").toStdString();
    EXPECT_EQ(gui->findChild<LogWindow *>(), nullptr)
        << "declining the warning ran the deck anyway";
    EXPECT_TRUE(finished.isEmpty());
}

TEST_F(Run, AQuitDeckIsNotEvenOfferedWhenTheWordIsPartOfSomethingElse)
{
    // "quitting" in a comment is not the quit command; the guard matches the
    // command at the start of a line, so this must run without asking
    Modals modals;
    setBuffer("# nothing about quitting here\n"
              "print \"requitted\"\n");
    runBuffer();
    EXPECT_FALSE(modals.said("terminate not only")) << "the quit guard fired on a comment";
}

TEST_F(Run, StartingASecondRunWhileOneIsGoingIsRefused)
{
    // two SPARTA runs in one process would share the library and trample each
    // other, so the second has to be turned away rather than queued
    Modals modals;
    QString deck = QString::fromLatin1(kDeck);
    deck.replace("run             5", "run             200000");
    setBuffer(deck);

    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    // doRun() builds the log window as its last act, so this waits until the
    // first run is genuinely under way rather than guessing at a delay
    ASSERT_TRUE(waitFor([this] { return gui->findChild<LogWindow *>() != nullptr; }, 10000))
        << "the first run never started";

    gui->runBuffer();
    waitFor([&modals] { return modals.count > 0; }, 5000);
    const bool refused = modals.said("Must stop current run");

    QMetaObject::invokeMethod(gui, "stopRun");
    waitFor([&finished] { return !finished.isEmpty(); }, 30000);

    EXPECT_TRUE(refused) << "a second run was started on top of a running one: "
                         << modals.seen.join(" | ").toStdString();
    EXPECT_EQ(finished.count(), 1) << "two runs finished when only one was allowed to start";
}

TEST_F(Run, StoppingARunEndsIt)
{
    Modals modals;
    // a longer run, so there is something to stop
    QString deck = QString::fromLatin1(kDeck);
    deck.replace("run             5", "run             200000");
    setBuffer(deck);

    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    waitFor([] { return false; }, 300); // let it get going
    QMetaObject::invokeMethod(gui, "stopRun");

    EXPECT_TRUE(waitFor([&finished] { return !finished.isEmpty(); }, 30000))
        << "the run did not end after being stopped";
}

// ---------------------------------------------------------------- archiving

TEST_F(Run, ArchivingIsOffByDefaultAndNothingIsRecorded)
{
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());

    auto *history = gui->findChild<RunHistory *>();
    if (history) EXPECT_EQ(history->count(), 0) << "a run was archived without being asked";
}

TEST_F(Run, AnArchivedRunRecordsTheDeckTheLogAndItsProvenance)
{
    QSettings settings;
    settings.setValue(Keys::ARCHIVE_RUNS, true);
    settings.sync();

    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());

    auto *history = gui->findChild<RunHistory *>();
    ASSERT_NE(history, nullptr) << "archiving was on but no history was created";
    ASSERT_EQ(history->count(), 1) << "the finished run was not archived";

    const auto &rec = history->at(0);
    EXPECT_EQ(rec.status, "ok");
    EXPECT_EQ(rec.deckName, "buffer") << "an unsaved buffer should archive under that name";
    EXPECT_TRUE(rec.deckText.contains("create_box")) << "the deck was not recorded";
    EXPECT_FALSE(rec.logText.isEmpty()) << "the log was not recorded";
    EXPECT_FALSE(rec.timestamp.isEmpty());
    EXPECT_FALSE(rec.id.isEmpty());

    // the provenance an archived run has to carry to be traceable at all
    for (const char *key : {"Run number", "SPARTA version", "Parallelism", "Accelerator",
                            "Packages", "I/O support", "Host", "OS", "Kernel", "Architecture",
                            "Command line"})
        EXPECT_TRUE(rec.metadata.contains(QLatin1String(key)))
            << "the record has no \"" << key << "\"";
}

TEST_F(Run, AnArchivedRunIsNamedAfterItsDeckWhenThereIsOne)
{
    QSettings settings;
    settings.setValue(Keys::ARCHIVE_RUNS, true);
    settings.sync();

    const QString deck = writeFile("in.archived", kDeck);
    Modals modals;
    gui->openFile(deck);
    ASSERT_TRUE(runBuffer());

    auto *history = gui->findChild<RunHistory *>();
    ASSERT_NE(history, nullptr);
    ASSERT_EQ(history->count(), 1);
    EXPECT_EQ(history->at(0).deckName, "in.archived");
    EXPECT_FALSE(history->at(0).metadata.value("Working directory").isEmpty());
}

TEST_F(Run, AFailedRunIsArchivedAsFailed)
{
    QSettings settings;
    settings.setValue(Keys::ARCHIVE_RUNS, true);
    settings.sync();

    Modals modals;
    setBuffer("this is not a SPARTA command\n");
    runBuffer();

    auto *history = gui->findChild<RunHistory *>();
    ASSERT_NE(history, nullptr);
    ASSERT_EQ(history->count(), 1) << "a failed run was not archived";
    EXPECT_EQ(history->at(0).status, "failed")
        << "a run that failed was archived as if it had succeeded";
}

TEST_F(Run, EachRunGetsItsOwnArchiveEntry)
{
    QSettings settings;
    settings.setValue(Keys::ARCHIVE_RUNS, true);
    settings.sync();

    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());
    ASSERT_TRUE(runBuffer());

    auto *history = gui->findChild<RunHistory *>();
    ASSERT_NE(history, nullptr);
    EXPECT_EQ(history->count(), 2);
    EXPECT_NE(history->at(0).id, history->at(1).id) << "two runs share one archive id";
    EXPECT_NE(history->at(0).metadata.value("Run number"),
              history->at(1).metadata.value("Run number"))
        << "the run counter did not advance between runs";
}

// ---------------------------------------------------------------- restart continuation

TEST_F(Run, ContinuingFromARestartInsertsTheCommandsForReview)
{
    // no simulator needed: this lists the restart files it can see and writes
    // a read_restart + run pair into the editor rather than running anything
    writeFile("tmp.restart.50", QByteArray(64, 'x'));
    setBuffer("");

    class Accept : public QObject {
    public:
        Accept()
        {
            timer.setInterval(10);
            connect(&timer, &QTimer::timeout, this, [this]() {
                if ((left -= 10) < 0) { timer.stop(); return; }
                auto *m = QApplication::activeModalWidget();
                if (!m) return;
                if (auto *list = m->findChild<QListWidget *>()) {
                    rows = list->count();
                    if (rows > 0) list->setCurrentRow(0);
                }
                if (auto *spin = m->findChild<QSpinBox *>()) spin->setValue(250);
                if (auto *d = qobject_cast<QDialog *>(m)) d->accept();
                ++seen;
                timer.stop();
            });
            timer.start();
        }
        int seen = 0;
        int rows = -1;

    private:
        QTimer timer;
        int left = 4000;
    } accept;

    QMetaObject::invokeMethod(gui, "continueRestart");
    QCoreApplication::processEvents();

    ASSERT_GE(accept.seen, 1) << "the restart dialog never appeared";
    EXPECT_GE(accept.rows, 1) << "the restart file in the working directory was not listed";

    const QString text = editor()->toPlainText();
    EXPECT_TRUE(text.contains("read_restart")) << text.toStdString();
    EXPECT_TRUE(text.contains("tmp.restart.50")) << text.toStdString();
    EXPECT_TRUE(text.contains("run 250")) << "the step count from the dialog was not used: "
                                          << text.toStdString();
}

TEST_F(Run, TheRestartListOffersOnlyRestartFiles)
{
    // a bare "*restart*" glob would offer log files and notes that merely
    // mention the word; the filters are deliberately narrow
    writeFile("tmp.restart.10", QByteArray(16, 'x'));
    writeFile("run.spart", QByteArray(16, 'x'));
    writeFile("notes-about-restart.txt", "not a restart file\n");
    writeFile("log.restart.txt", "also not one\n");

    class Count : public QObject {
    public:
        Count()
        {
            timer.setInterval(10);
            connect(&timer, &QTimer::timeout, this, [this]() {
                if ((left -= 10) < 0) { timer.stop(); return; }
                auto *m = QApplication::activeModalWidget();
                if (!m) return;
                if (auto *list = m->findChild<QListWidget *>()) {
                    for (int i = 0; i < list->count(); ++i)
                        names << list->item(i)->text();
                }
                if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
                timer.stop();
            });
            timer.start();
        }
        QStringList names;

    private:
        QTimer timer;
        int left = 4000;
    } counter;

    QMetaObject::invokeMethod(gui, "continueRestart");
    QCoreApplication::processEvents();

    const QString all = counter.names.join(" | ");
    EXPECT_TRUE(all.contains("tmp.restart.10")) << all.toStdString();
    EXPECT_TRUE(all.contains("run.spart")) << all.toStdString();
    EXPECT_FALSE(all.contains("notes-about-restart.txt"))
        << "a text file that merely mentions restart was offered: " << all.toStdString();
}

TEST_F(Run, CancellingTheRestartDialogWritesNothing)
{
    writeFile("tmp.restart.99", QByteArray(16, 'x'));
    setBuffer("untouched\n");

    Modals modals; // rejects every dialog
    QMetaObject::invokeMethod(gui, "continueRestart");
    QCoreApplication::processEvents();
    EXPECT_EQ(editor()->toPlainText(), "untouched\n");
}

TEST_F(Run, TheRestartDialogWithNothingToOfferSaysSoRatherThanInserting)
{
    setBuffer("untouched\n");

    // accept with an empty list: there is nothing to select
    class AcceptEmpty : public QObject {
    public:
        AcceptEmpty()
        {
            timer.setInterval(10);
            connect(&timer, &QTimer::timeout, this, [this]() {
                if ((left -= 10) < 0) { timer.stop(); return; }
                auto *m = QApplication::activeModalWidget();
                if (!m) return;
                if (auto *box = qobject_cast<QMessageBox *>(m)) {
                    said << box->text();
                    box->accept();
                    return;
                }
                if (auto *d = qobject_cast<QDialog *>(m)) d->accept();
            });
            timer.start();
        }
        QStringList said;

    private:
        QTimer timer;
        int left = 4000;
    } accept;

    QMetaObject::invokeMethod(gui, "continueRestart");
    QCoreApplication::processEvents();

    EXPECT_EQ(editor()->toPlainText(), "untouched\n")
        << "commands were inserted with no restart file selected";
    EXPECT_TRUE(accept.said.join(" ").contains("No restart file selected"))
        << accept.said.join(" | ").toStdString();
}

// ---------------------------------------------------------------- 3D snapshot
//
// Only built with VTK: renderVtkSnapshot() and the SceneWindow it opens are
// inside #if SPARTA_GUI_HAVE_VTK, so without it the slot does not exist.
// VTK renders through its own X connection even when Qt is offscreen, which is
// why this suite runs under Xvfb.
#if defined(SPARTA_GUI_HAVE_VTK)


TEST_F(Run, TheVtkSnapshotIsRefusedWhileARunIsGoing)
{
    Modals modals;
    QString deck = QString::fromLatin1(kDeck);
    deck.replace("run             5", "run             200000");
    setBuffer(deck);

    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    waitFor([] { return false; }, 300);

    QMetaObject::invokeMethod(gui, "renderVtkSnapshot");
    QCoreApplication::processEvents();
    const bool refused = modals.said("while SPARTA is running");

    QMetaObject::invokeMethod(gui, "stopRun");
    waitFor([&finished] { return !finished.isEmpty(); }, 30000);

    EXPECT_TRUE(refused) << "a 3D snapshot was attempted mid-run: "
                         << modals.seen.join(" | ").toStdString();
}

TEST_F(Run, TheVtkSnapshotSaysWhenTheLibraryCannotWriteVtk)
{
    // A stock SPARTA has no VTK package. The window still opens the viewer, so
    // files written elsewhere can be loaded -- saying so rather than failing
    // silently is the whole of this branch.
    Modals modals;
    setBuffer(QString::fromLatin1(kDeck));
    ASSERT_TRUE(runBuffer());

    QMetaObject::invokeMethod(gui, "renderVtkSnapshot");
    QCoreApplication::processEvents();
    waitFor([&modals] { return modals.count > 0; }, 5000);

    if (modals.said("without the VTK package")) {
        SUCCEED() << "the library has no VTK dump styles and the window said so";
    } else {
        // a VTK-enabled library takes the other branch and writes a snapshot
        EXPECT_FALSE(modals.said("Cannot create a 3D snapshot"))
            << modals.seen.join(" | ").toStdString();
    }
}

TEST_F(Run, TheVtkSnapshotRefusesADeckThatMakesNoBox)
{
    Modals modals;
    setBuffer("# a deck with no create_box in it at all\nprint \"hello\"\n");

    QMetaObject::invokeMethod(gui, "renderVtkSnapshot");
    QCoreApplication::processEvents();
    waitFor([&modals] { return modals.count > 0; }, 8000);

    // either it has no VTK styles (reported first) or it refuses for want of a
    // box; both are a refusal rather than an empty window
    EXPECT_GE(modals.count, 1) << "a snapshot was attempted with no system box";
}

#endif // SPARTA_GUI_HAVE_VTK

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
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

    // Settings and an archive directory of this process's own: the run archive
    // is written under AppDataLocation, and a case that clears the settings
    // must not wipe what a concurrent process just wrote.
    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_run-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
