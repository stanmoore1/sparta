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

// Inspecting a restart file: SpartaGui::inspectFile() and purgeInspectList().
//
// This is the one place the application opens somebody's saved simulation
// without running it -- point it at a restart file and it tells you what is
// inside and draws you a picture.  It had no coverage at all, and it is not a
// read-only path: it loads the file into the live SPARTA instance, replacing
// whatever was there, and writes a temporary log beside the user's file.
//
// So the checks here are as much about what it does not do -- to a file that is
// not a restart, to the instance when the read fails, to the disk afterwards --
// as about the windows it opens.

#include "spartagui.h"

#include "codeeditor.h"
#include "constants.h"
#include "fileviewer.h"
#include "helpers.h"

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QIcon>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTextEdit>
#include <QTimer>

#include <memory>

namespace {

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

/// Answers whatever modal appears and records it.
class Modals : public QObject {
public:
    explicit Modals(QMessageBox::StandardButton button = QMessageBox::Yes, int budgetMs = 60000) :
        button(button), left(budgetMs)
    {
        timer.setInterval(10);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }
    QStringList seen;
    int boxes = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : seen)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return seen.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 10) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            ++boxes;
            seen << box->text() + " " + box->informativeText() + " " + box->detailedText();
            if (auto *b = box->button(button)) b->click();
            else box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) static_cast<QDialog *>(fd)->reject();
    }
    QTimer timer;
    QMessageBox::StandardButton button;
    int left;
};

/// inspectFile is protected; the Run menu reaches it through the file dialog.
class TestableGui : public SpartaGui {
public:
    using SpartaGui::inspectFile;
    using SpartaGui::SpartaGui;
};

// A deck that writes a restart file with a known, checkable shape: 64 grid
// cells and 50 particles, so the summary it produces can be read back.
const char *const kDeck = "seed 12345\n"
                          "dimension 3\n"
                          "global gridcut 0.0 comm/sort yes\n"
                          "boundary r r r\n"
                          "create_box 0 10 0 10 0 10\n"
                          "create_grid 4 4 4\n"
                          "species ar.species Ar\n"
                          "mixture air Ar vstream 0.0 0.0 0.0\n"
                          "global nrho 1.0 fnum 1.0\n"
                          "create_particles air n 50\n"
                          "collide vss air ar.vss\n"
                          "run 0\n"
                          "write_restart saved.restart\n";

const char *const kSpecies = "# ID, molwt, molmass, rotdof, rotrel, vibdof, vibrel, vibtemp, wt, q\n"
                             "Ar  40.00    6.63E-26  0    .0   0   .0    0.0    1.0      0.0\n";
const char *const kVss     = "# diameter, omega, tref, alpha\n"
                             "Ar   4.11e-10 0.81  273.15  1.4\n";

class Inspect : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!*testLibrary()) GTEST_SKIP() << "no shared libsparta";
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.setValue(Keys::RESTORE_SESSION, false);
        settings.sync();

        write("ar.species", kSpecies);
        write("ar.vss", kVss);
        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());

        gui = new TestableGui(nullptr, QString(), 800, 600);
    }

    void TearDown() override
    {
        closeExtraWindows();
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    void write(const QString &name, const QString &text) const
    {
        QFile f(dir.filePath(name));
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
    }

    static void closeExtraWindows()
    {
        for (auto *w : QApplication::topLevelWidgets())
            if (!qobject_cast<SpartaGui *>(w) && w->isVisible()) w->close();
        QApplication::processEvents();
    }

    template <class F> static bool waitFor(F done, int budgetMs)
    {
        QElapsedTimer clock;
        clock.start();
        while (!done() && clock.elapsed() < budgetMs)
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        return done();
    }

    /// Run the deck once so there is a real restart file to inspect.
    QString makeRestart()
    {
        gui->findChild<CodeEditor *>()->setPlainText(QString::fromLatin1(kDeck));
        QSignalSpy finished(gui, &SpartaGui::runFinished);
        gui->runBuffer();
        EXPECT_TRUE(waitFor([&finished] { return !finished.isEmpty(); }, 120000))
            << "the deck that writes the restart file never finished";
        const QString p = dir.filePath("saved.restart");
        EXPECT_TRUE(QFile::exists(p)) << "the run wrote no restart file";
        return p;
    }

    /// The read-only viewers inspectFile opens, by title.
    static QStringList viewerTitles()
    {
        QStringList out;
        for (auto *w : QApplication::topLevelWidgets())
            if (qobject_cast<FileViewer *>(w)) out << w->windowTitle();
        return out;
    }

    static int viewerCount()
    {
        int n = 0;
        for (auto *w : QApplication::topLevelWidgets())
            if (qobject_cast<FileViewer *>(w)) ++n;
        return n;
    }

    /// Everything a FileViewer is showing, so the summary can be read.
    static QString viewerText()
    {
        // FileViewer *is* the text widget, so it is the window itself that holds
        // the text rather than a child of it
        QString all;
        for (auto *w : QApplication::topLevelWidgets())
            if (auto *v = qobject_cast<FileViewer *>(w)) all += v->toPlainText();
        return all;
    }

    QTemporaryDir dir;
    QString startDir;
    TestableGui *gui = nullptr;
};

} // namespace

// ---------------------------------------------------------------- refusals

TEST_F(Inspect, AnEmptyNameDoesNothing)
{
    Modals modals;
    gui->inspectFile(QString());
    EXPECT_EQ(modals.boxes, 0) << modals.all().toStdString();
    EXPECT_EQ(viewerCount(), 0);
}

TEST_F(Inspect, AFileThatIsNotARestartFileIsRefusedBySignatureNotByName)
{
    // named like one and nothing like one inside: reading it as a restart would
    // wipe the live instance and leave the user with an empty simulation
    Modals modals;
    write("fake.restart", "I am not a SPARTA restart file, whatever my name says.\n");
    gui->inspectFile(dir.filePath("fake.restart"));

    EXPECT_TRUE(modals.said("is not a SPARTA restart file")) << modals.all().toStdString();
    EXPECT_EQ(viewerCount(), 0) << "it opened a window for a file it rejected";
}

TEST_F(Inspect, AFileThatCannotBeOpenedIsReported)
{
    Modals modals;
    QDir(dir.path()).mkdir("adirectory.restart");
    gui->inspectFile(dir.filePath("adirectory.restart"));

    EXPECT_EQ(viewerCount(), 0);
    EXPECT_TRUE(modals.said("Cannot open file") || modals.said("is not a SPARTA restart file"))
        << modals.all().toStdString();
}

TEST_F(Inspect, AnInputDeckIsNotMistakenForARestart)
{
    // the two live side by side in a working directory and the dialog offers
    // both; a text deck must be refused rather than half-read
    Modals modals;
    write("in.deck", QString::fromLatin1(kDeck));
    gui->inspectFile(dir.filePath("in.deck"));
    EXPECT_TRUE(modals.said("is not a SPARTA restart file")) << modals.all().toStdString();
    EXPECT_EQ(viewerCount(), 0);
}

// ------------------------------------------------------------- the real thing

TEST_F(Inspect, ShowsWhatIsInsideTheRestartFile)
{
    Modals modals;
    const QString restart = makeRestart();
    ASSERT_FALSE(restart.isEmpty());
    closeExtraWindows(); // the run may have opened panels of its own

    gui->inspectFile(restart);
    QApplication::processEvents();

    ASSERT_GE(viewerCount(), 1) << "no window opened for a valid restart file";
    EXPECT_TRUE(viewerTitles().join(" ").contains("saved.restart"))
        << "the window does not say which file it is showing: "
        << viewerTitles().join(", ").toStdString();

    // SPARTA's own read_restart summary, which is the whole point of the window
    const QString text = viewerText();
    EXPECT_TRUE(text.contains("grid cells") || text.contains("child grid cells"))
        << "the summary does not describe the grid:\n"
        << text.left(600).toStdString();
    EXPECT_TRUE(text.contains("particles")) << text.left(600).toStdString();
}

TEST_F(Inspect, TheTemporaryLogItWritesDoesNotSurvive)
{
    // it writes <file>.info.log next to the user's restart file to hand to the
    // viewer; leaving it there litters their working directory
    Modals modals;
    const QString restart = makeRestart();
    closeExtraWindows();

    gui->inspectFile(restart);
    QApplication::processEvents();
    EXPECT_FALSE(QFile::exists(restart + ".info.log"))
        << "the temporary summary was left beside the user's file";
}

TEST_F(Inspect, WindowsTheUserStillHasOpenSurviveASecondInspection)
{
    // two restarts can be compared side by side, so an inspection that is still
    // on screen is not swept away by the next one
    Modals modals;
    const QString restart = makeRestart();
    closeExtraWindows();

    gui->inspectFile(restart);
    QApplication::processEvents();
    ASSERT_GE(viewerCount(), 1);

    gui->inspectFile(restart);
    QApplication::processEvents();
    EXPECT_GE(viewerCount(), 2) << "the first inspection was closed behind the user's back";
}

TEST_F(Inspect, WindowsTheUserClosedAreCollectedByTheNextInspection)
{
    // and the ones they did close are reclaimed rather than accumulating: the
    // list is purged of everything already hidden each time round
    Modals modals;
    const QString restart = makeRestart();
    closeExtraWindows();

    gui->inspectFile(restart);
    QApplication::processEvents();
    const int one = viewerCount();
    ASSERT_GE(one, 1);

    closeExtraWindows(); // the user closes them
    gui->inspectFile(restart);
    QApplication::processEvents();
    EXPECT_EQ(viewerCount(), one)
        << "closed inspection windows were kept and a second set added on top";
}

TEST_F(Inspect, TheWindowIsUsableAgainAfterwards)
{
    // inspectFile() clears the live instance and reads the restart into it; the
    // editor and the next run have to survive that
    Modals modals;
    const QString restart = makeRestart();
    closeExtraWindows();
    gui->inspectFile(restart);
    QApplication::processEvents();

    gui->findChild<CodeEditor *>()->setPlainText(QString::fromLatin1(kDeck));
    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    ASSERT_TRUE(waitFor([&finished] { return !finished.isEmpty(); }, 120000))
        << "a run after an inspection never finished";
    EXPECT_TRUE(finished.at(0).at(0).toBool())
        << "the run after an inspection failed; the instance was left unusable";
}

TEST_F(Inspect, ARestartWithNoSeedStillRendersBecauseOneIsSupplied)
{
    // read_restart does not restore the RNG seed, and rendering runs "run 0",
    // which a deck with a collide style refuses without one.  inspectFile
    // supplies a fixed seed for exactly this reason.
    Modals modals;
    const QString restart = makeRestart();
    closeExtraWindows();

    gui->inspectFile(restart);
    QApplication::processEvents();

    EXPECT_FALSE(modals.said("Error reading restart file"))
        << "the restart could not be read back: " << modals.all().toStdString();
    EXPECT_FALSE(modals.said("Seed command has not been used"))
        << "rendering the restored state hit the missing-seed error: "
        << modals.all().toStdString();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT  = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_inspect-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
