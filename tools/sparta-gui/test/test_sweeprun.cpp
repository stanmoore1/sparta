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

// The parametric sweep driver, and the library boundary it reads its numbers
// through.
//
// SweepController runs unattended over many combinations and hands back a table
// the user reads as science.  Nothing in it was covered: the panel around it has
// tests, but the driver needs a live SpartaGui and a running simulator, and
// never got one.  That matters more than the line count suggests, because its
// failures are quiet -- a wrong keyword-to-row match in readThermo(), or an
// off-by-one in onRunFinished()'s cursor, yields a table that is entirely
// self-consistent and entirely wrong.
//
// So every assertion here is against an arithmetic answer.  The deck creates
// exactly ${n} particles, so Np is n, and a sweep over n has a table that can be
// checked digit for digit rather than merely for being present.

#include "spartagui.h"

#include "codeeditor.h"
#include "constants.h"
#include "spartawrapper.h"
#include "sweeppanel.h"
#include "sweepspec.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QTableWidget>
#include <QSpinBox>
#include <QRadioButton>
#include <QPushButton>
#include <QProgressBar>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFont>
#include <QIcon>
#include <QMessageBox>
#include <QSettings>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>
#include <memory>

using namespace Sweep;

namespace {

// Exactly ${n} particles, so Np is n and the sweep's answer is arithmetic.
// "run 0" keeps each combination to well under a second.
const char *const kSweepDeck = "seed            12345\n"
                               "dimension       2\n"
                               "global          gridcut 0.0 comm/sort yes\n"
                               "boundary        o r p\n"
                               "create_box      0 10 0 10 -0.5 0.5\n"
                               "create_grid     4 4 1\n"
                               "species         ar.species Ar\n"
                               "mixture         air Ar vstream 0.0 0.0 0.0\n"
                               "global          nrho 1.0 fnum 1.0\n"
                               "create_particles air n ${n}\n"
                               "collide         vss air ar.vss\n"
                               "stats           1\n"
                               "run             0\n";

// Same system, but reflective on every side so no particle ever leaves, and long
// enough that the 10 ms stats poller collects a series rather than a single
// point.  Np is conserved, so min, max, mean and final all have to agree on n --
// which is what makes a reducer that samples the wrong column visible.
const char *const kSampledDeck = "seed            12345\n"
                                 "dimension       2\n"
                                 "global          gridcut 0.0 comm/sort yes\n"
                                 "boundary        r r p\n"
                                 "create_box      0 10 0 10 -0.5 0.5\n"
                                 "create_grid     4 4 1\n"
                                 "species         ar.species Ar\n"
                                 "mixture         air Ar vstream 0.0 0.0 0.0\n"
                                 "global          nrho 1.0 fnum 1.0\n"
                                 "create_particles air n ${n}\n"
                                 "collide         vss air ar.vss\n"
                                 "stats           1000\n"
                                 "run             10000\n";

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

// Dismisses anything modal. A sweep that puts up a dialog would otherwise stall
// with nobody to answer it.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 120000) : left(budgetMs)
    {
        timer.setInterval(10);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }
    QStringList seen;
    int count = 0;

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 10) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        ++count;
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            seen << box->text();
            box->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }
    QTimer timer;
    int left;
};

class Sweeping : public ::testing::Test {
protected:
    void SetUp() override
    {
        REQUIRE_LIBRARY();
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.sync();

        write("ar.species", kSpecies);
        write("ar.vss", kVss);
        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());

        gui = new SpartaGui(nullptr, QString(), 800, 600);
        gui->findChild<CodeEditor *>()->setPlainText(QString::fromLatin1(kSweepDeck));
    }

    void TearDown() override
    {
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    void write(const QString &name, const QByteArray &bytes) const
    {
        QFile f(dir.filePath(name));
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
    }

    template <class F> static bool waitFor(F done, int budgetMs)
    {
        QElapsedTimer clock;
        clock.start();
        while (!done() && clock.elapsed() < budgetMs)
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        return done();
    }

    /// The application's own sweep panel, built the way the Run menu builds it.
    /// The controller and the wrapper it drives are private to the window, so
    /// everything here goes through the panel a user actually operates.
    SweepPanel *panel()
    {
        if (!gui->findChild<SweepPanel *>()) QMetaObject::invokeMethod(gui, "runSweep");
        auto *p = gui->findChild<SweepPanel *>();
        EXPECT_NE(p, nullptr) << "the Run menu did not build a sweep panel";
        return p;
    }

    SweepResultsModel *results() { return panel()->findChild<SweepResultsModel *>(); }

    void useDeck(const char *deck)
    {
        gui->findChild<CodeEditor *>()->setPlainText(QString::fromLatin1(deck));
    }

    template <class W> W *ctl(const char *name) { return panel()->findChild<W *>(name); }

    /// Fill in a variable row, the way a user would.  The panel starts with one
    /// blank row already (it offers the first variable it discovered in the
    /// buffer), so fill that one before adding more -- a row left blank is a
    /// spec error, not an empty sweep.
    void addVariable(const QString &name, const QString &type, const QString &spec)
    {
        auto *table = ctl<QTableWidget>("varTable");
        if (!table) {
            ADD_FAILURE() << "the sweep panel has no variable table";
            return;
        }
        int r = -1;
        for (int i = 0; i < table->rowCount() && r < 0; ++i)
            if (table->item(i, 2) && table->item(i, 2)->text().trimmed().isEmpty()) r = i;
        if (r < 0) {
            QMetaObject::invokeMethod(panel(), "addVariableRow");
            r = table->rowCount() - 1;
        }
        qobject_cast<QComboBox *>(table->cellWidget(r, 0))->setCurrentText(name);
        qobject_cast<QComboBox *>(table->cellWidget(r, 1))->setCurrentText(type);
        table->item(r, 2)->setText(spec);
    }

    void setQuantities(const QString &csv) { ctl<QLineEdit>("quantities")->setText(csv); }

    void setReducer(Reducer r)
    {
        auto *box = ctl<QComboBox>("reducer");
        box->setCurrentIndex(box->findData(static_cast<int>(r)));
    }

    void setReplicates(int n, const QString &seedVar = QString(), int seedBase = -1)
    {
        ctl<QSpinBox>("replicates")->setValue(n);
        if (!seedVar.isEmpty()) ctl<QLineEdit>("seedVar")->setText(seedVar);
        if (seedBase >= 0) ctl<QSpinBox>("seedBase")->setValue(seedBase);
    }

    void setZip() { ctl<QRadioButton>("zip")->setChecked(true); }

    /// press Run Sweep and wait for the panel to say it is done
    bool runSweep(int budgetMs = 180000)
    {
        auto *btn    = ctl<QPushButton>("startSweep");
        auto *status = statusLabel();
        if (!btn || !status) {
            ADD_FAILURE() << "the sweep panel has no start button or status line";
            return false;
        }
        QMetaObject::invokeMethod(panel(), "startSweep");

        // startSweep() flips the button synchronously once the spec is accepted,
        // so anything still reading "Run Sweep" here was refused outright -- and
        // the reason went into a message box the reaper has already dismissed
        if (btn->text() != QLatin1String("Stop Sweep")) {
            ADD_FAILURE() << "the sweep was refused before it started: "
                          << (modals.seen.isEmpty() ? QString("(no dialog)")
                                                    : modals.seen.join(" | "))
                                 .toStdString();
            return false;
        }

        const bool done =
            waitFor([btn] { return btn->text() == QLatin1String("Run Sweep"); }, budgetMs);
        EXPECT_TRUE(done) << "the sweep did not finish within " << budgetMs << " ms";
        EXPECT_TRUE(status->text().startsWith("Sweep complete"))
            << "the sweep ended saying: " << status->text().toStdString();
        return done && status->text().startsWith("Sweep complete");
    }

    /// the panel's status line, which says whether the sweep completed
    QLabel *statusLabel() { return ctl<QLabel>("status"); }

    QString cell(int row, int col)
    {
        auto *m = results();
        return m ? m->data(m->index(row, col), Qt::DisplayRole).toString() : QString();
    }

    QTemporaryDir dir;
    QString startDir;
    SpartaGui *gui = nullptr;
    Modals modals; ///< nothing here expects a dialog; one appearing is the finding
};

} // namespace

// ---------------------------------------------------------------- the table

TEST_F(Sweeping, TabulatesOneRowPerCombinationInOrderWithTheRightNumbers)
{
    // the whole point of the feature: n particles in, n reported back, one row
    // per value, in the order the values were given
    addVariable("n", "List", "40, 80, 120");
    setQuantities("Np");
    ASSERT_TRUE(runSweep());

    auto *m = results();
    ASSERT_NE(m, nullptr);
    ASSERT_EQ(m->rowCount(), 3) << "one row per swept value";
    ASSERT_EQ(m->columnCount(), 2) << "the variable and the quantity";
    const QStringList want{"40", "80", "120"};
    for (int r = 0; r < 3; ++r) {
        EXPECT_EQ(cell(r, 0), want.at(r)) << "row " << r << " is not the combination it claims";
        EXPECT_EQ(cell(r, 1), want.at(r))
            << "row " << r << " tabulated " << cell(r, 1).toStdString()
            << " particles for n=" << want.at(r).toStdString();
    }
}

TEST_F(Sweeping, TheHeadersNameTheVariableAndTheQuantityWithItsReducer)
{
    addVariable("n", "List", "40");
    setQuantities("Np");
    setReducer(Reducer::Final);
    ASSERT_TRUE(runSweep());

    const QStringList h = results()->headers();
    ASSERT_EQ(h.size(), 2);
    EXPECT_EQ(h.at(0), "n");
    EXPECT_TRUE(h.at(1).startsWith("Np (")) << h.at(1).toStdString();
    EXPECT_TRUE(h.at(1).contains(reducerName(Reducer::Final))) << h.at(1).toStdString();
}

TEST_F(Sweeping, ACartesianSweepCoversEveryPairInOrder)
{
    // every combination exactly once, with the second variable varying fastest
    addVariable("n", "List", "40, 80");
    addVariable("unused", "List", "1, 2");
    setQuantities("Np");
    ASSERT_TRUE(runSweep());

    ASSERT_EQ(results()->rowCount(), 4);
    QStringList pairs;
    for (int r = 0; r < 4; ++r) {
        pairs << cell(r, 0) + "/" + cell(r, 1);
        EXPECT_EQ(cell(r, 2), cell(r, 0)) << "row " << r << " tabulated the wrong count";
    }
    EXPECT_EQ(pairs, (QStringList{"40/1", "40/2", "80/1", "80/2"}))
        << pairs.join(", ").toStdString();
}

TEST_F(Sweeping, AZippedSweepPairsTheValuesRatherThanCrossingThem)
{
    addVariable("n", "List", "40, 80");
    addVariable("unused", "List", "1, 2");
    setQuantities("Np");
    setZip();
    ASSERT_TRUE(runSweep());

    EXPECT_EQ(results()->rowCount(), 2) << "a zip should not produce the cartesian product";
    EXPECT_EQ(cell(0, 0) + "/" + cell(0, 1), "40/1");
    EXPECT_EQ(cell(1, 0) + "/" + cell(1, 1), "80/2");
}

TEST_F(Sweeping, AQuantityTheRunDoesNotProduceIsReportedAsUnavailable)
{
    // readThermo() returns NaN for a keyword the run never printed; the cell
    // has to say so rather than show a zero that looks like a measurement
    addVariable("n", "List", "40");
    setQuantities("NoSuchQuantity");
    ASSERT_TRUE(runSweep());

    ASSERT_EQ(results()->rowCount(), 1);
    EXPECT_EQ(cell(0, 1), "n/a")
        << "a quantity the run never produced was tabulated as " << cell(0, 1).toStdString();
}

TEST_F(Sweeping, TwoQuantitiesGetAColumnEach)
{
    addVariable("n", "List", "40, 80");
    setQuantities("Np, Natt");
    ASSERT_TRUE(runSweep());

    EXPECT_EQ(results()->columnCount(), 3) << "the variable plus one column per quantity";
    for (int r = 0; r < results()->rowCount(); ++r)
        EXPECT_EQ(cell(r, 1), cell(r, 0)) << "the Np column moved";
}

// ---------------------------------------------------------------- reducers

TEST_F(Sweeping, EveryReducerProducesTheSameAnswerOnAConstantSeries)
{
    // Everything but "final" reduces the series the stats poller collected
    // during the run, so this needs a run long enough to have one.  The box is
    // closed, so Np is conserved and all four reducers have to land on n: a
    // reducer reading the wrong column, or an empty series quietly reduced to
    // zero, shows up immediately.
    useDeck(kSampledDeck);
    addVariable("n", "List", "60");
    setQuantities("Np");
    for (auto red : {Reducer::Final, Reducer::Min, Reducer::Max, Reducer::Mean}) {
        setReducer(red);
        ASSERT_TRUE(runSweep()) << "reducer " << reducerName(red).toStdString();
        ASSERT_EQ(results()->rowCount(), 1);
        EXPECT_EQ(cell(0, 1), "60")
            << reducerName(red).toStdString() << " tabulated " << cell(0, 1).toStdString();
        EXPECT_TRUE(results()->headers().at(1).contains(reducerName(red)))
            << "the header does not name the reducer used";
    }
}

TEST_F(Sweeping, AReducerWithNothingToReduceReportsUnavailableRatherThanZero)
{
    // "run 0" is over before the stats poller ever ticks, so min has no series
    // to reduce.  That absence has to reach the table as "n/a": a 0 would sit
    // there looking like a measurement of an empty box.
    addVariable("n", "List", "40");
    setQuantities("Np");
    setReducer(Reducer::Min);
    ASSERT_TRUE(runSweep());

    ASSERT_EQ(results()->rowCount(), 1);
    EXPECT_EQ(cell(0, 1), "n/a")
        << "a run that produced no samples was tabulated as " << cell(0, 1).toStdString();
}

// ---------------------------------------------------------------- replicates

TEST_F(Sweeping, ReplicatesProduceOneRowWithAMeanAndAStandardError)
{
    addVariable("n", "List", "40, 80");
    setQuantities("Np");
    setReplicates(3, "sd", 100);
    ASSERT_TRUE(runSweep());

    EXPECT_EQ(results()->rowCount(), 2) << "three replicates of two points is still two rows";
    ASSERT_EQ(results()->columnCount(), 3) << "the variable, the mean and the standard error";

    const QStringList h = results()->headers();
    EXPECT_TRUE(h.at(1).endsWith(" mean")) << h.at(1).toStdString();
    EXPECT_TRUE(h.at(2).endsWith(" +/-SE")) << h.at(2).toStdString();

    // Np is n every replicate, so the mean is n and the spread is zero
    for (int r = 0; r < 2; ++r) {
        EXPECT_EQ(cell(r, 1), cell(r, 0))
            << "the mean of three identical runs is not the value itself";
        EXPECT_DOUBLE_EQ(cell(r, 2).toDouble(), 0.0)
            << "identical replicates produced a non-zero standard error: "
            << cell(r, 2).toStdString();
    }
}

TEST_F(Sweeping, EachReplicateRunsWithADistinctSeed)
{
    // The reason replicates exist: same inputs, a different random stream each
    // time.  A repeated seed makes an ensemble of identical runs and an error
    // bar that means nothing -- and it would look perfectly healthy in the
    // table, because every cell would still be filled in.
    //
    // The seed itself leaves no trace in the output, so this names the *particle
    // count* as the seed variable.  The deck creates ${n} particles, so replicate
    // k creates 40+k of them and the ensemble reads back 40, 41, 42, 43: mean
    // 41.5, sample sd sqrt(5/3), standard error half of that.  Repeat one seed
    // and none of those three numbers survives.
    addVariable("unused", "List", "1");
    setQuantities("Np");
    setReplicates(4, "n", 40);
    ASSERT_TRUE(runSweep());

    ASSERT_EQ(results()->rowCount(), 1);
    EXPECT_DOUBLE_EQ(cell(0, 1).toDouble(), 41.5)
        << "the replicates did not run with seeds 40..43; their mean was " << cell(0, 1).toStdString();
    EXPECT_NEAR(cell(0, 2).toDouble(), std::sqrt(5.0 / 3.0) / 2.0, 1e-6)
        << "the spread across replicates was " << cell(0, 2).toStdString();
}

TEST_F(Sweeping, ASingleReplicateNeedsNoSeedAndGetsNoErrorColumn)
{
    addVariable("n", "List", "40");
    setQuantities("Np");
    setReplicates(1);
    ASSERT_TRUE(runSweep());

    EXPECT_EQ(results()->columnCount(), 2) << "a single run got a standard-error column";
    EXPECT_FALSE(results()->headers().at(1).contains("mean"))
        << "a single run was labelled as an ensemble mean";
}

// ---------------------------------------------------------------- progress and control

TEST_F(Sweeping, ProgressCountsEveryRunNotEveryPoint)
{
    addVariable("n", "List", "40, 80");
    setQuantities("Np");
    setReplicates(2, "sd", 5);

    auto *bar = ctl<QProgressBar>("progress");
    ASSERT_NE(bar, nullptr);

    // the bar is the only sign of life during an unattended sweep, so it has to
    // move -- a correct range that never advances is the same as no bar at all
    QList<int> steps;
    auto c = QObject::connect(bar, &QProgressBar::valueChanged,
                              bar, [&steps](int v) { steps << v; });
    ASSERT_TRUE(runSweep());
    QObject::disconnect(c);

    EXPECT_EQ(bar->maximum(), 4)
        << "two points of two replicates is four runs, not " << bar->maximum();
    ASSERT_FALSE(steps.isEmpty()) << "the progress bar never moved during the sweep";
    for (int i = 1; i < steps.size(); ++i)
        EXPECT_GE(steps.at(i), steps.at(i - 1)) << "progress went backwards at step " << i;
    EXPECT_EQ(steps.last(), 4) << "the sweep ended with the bar short of its own total";

    // one tick as each of the four runs starts: a bar that advances per sweep
    // point instead stalls for the length of every replicate set
    QStringList seen;
    for (int v : steps) seen << QString::number(v);
    for (int run = 0; run < 4; ++run)
        EXPECT_TRUE(steps.contains(run))
            << "the bar never reported run " << run << "; it went " << seen.join(",").toStdString();
}

TEST_F(Sweeping, StoppingPartWayEndsTheSweepAndSaysItWasStopped)
{
    // a deck slow enough that "stop after the first row" is decisive rather
    // than a race with the remaining points finishing on their own
    useDeck(kSampledDeck);
    addVariable("n", "List", "40, 80, 120, 160, 200");
    setQuantities("Np");

    auto *btn = ctl<QPushButton>("startSweep");
    QMetaObject::invokeMethod(panel(), "startSweep");
    // let one point complete, then press the same button again to stop
    waitFor([this] { return results() && results()->rowCount() >= 1; }, 60000);
    QMetaObject::invokeMethod(panel(), "startSweep");

    ASSERT_TRUE(waitFor([btn] { return btn->text() == QLatin1String("Run Sweep"); }, 60000))
        << "the sweep did not end after being stopped";
    EXPECT_TRUE(statusLabel()->text().startsWith("Sweep stopped"))
        << "a stopped sweep reported: " << statusLabel()->text().toStdString();
    EXPECT_LT(results()->rowCount(), 5) << "the sweep ran to the end despite being stopped";
}

TEST_F(Sweeping, TheWindowIsUsableAgainAfterASweep)
{
    // the controller connects to the window's run signals and has to let go of
    // them, or the next ordinary run is still driving a finished sweep
    addVariable("n", "List", "40");
    setQuantities("Np");
    ASSERT_TRUE(runSweep());
    const int rows = results()->rowCount();

    gui->setRunVariables({{"n", "55"}});
    QSignalSpy finished(gui, &SpartaGui::runFinished);
    gui->runBuffer();
    ASSERT_TRUE(waitFor([&finished] { return !finished.isEmpty(); }, 60000));
    EXPECT_EQ(results()->rowCount(), rows)
        << "an ordinary run after the sweep added a results row";
}

TEST_F(Sweeping, TheResultsSurviveAndCanBeSweptAgain)
{
    addVariable("n", "List", "40");
    setQuantities("Np");
    ASSERT_TRUE(runSweep());
    ASSERT_EQ(results()->rowCount(), 1);

    // a second sweep replaces the table rather than appending to it
    ASSERT_TRUE(runSweep());
    EXPECT_EQ(results()->rowCount(), 1)
        << "the second sweep appended to the first one's results";
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
        QString("test_sweeprun-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
