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

// The per-surface extraction path: SpartaWrapper::extractCompute/extractFix.
//
// These are the only wrapper entry points that hand back a live pointer into
// SPARTA's own memory, and the surface report is their one consumer.  The
// reduction core (surfreport.cpp) has unit tests against hand-written arrays,
// but nothing had ever read a real one: a wrong style constant, a row/column
// transposition, or a stride mismatch between what SPARTA lays out and what the
// dialog walks would produce a report full of plausible numbers.
//
// So the checks here are self-referential on purpose.  The CSV export writes
// exactly the per-element array that was read, so summing a column of the CSV
// and comparing it to the integrated total the report prints closes the loop
// through both the library read and the reduction, and no assertion depends on
// a DSMC answer being any particular value.

#include "surfreportdialog.h"

#include "spartawrapper.h"
#include "surfreport.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QTemporaryDir>
#include <QTextStream>
#include <QTimer>

#include <cmath>

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

QString fixtures()
{
#if defined(SPARTA_SURF_FIXTURES)
    return QString(SPARTA_SURF_FIXTURES);
#else
    return QString();
#endif
}

// Answers the CSV export's save dialog with a path of our choosing.
class SaveTo : public QObject {
public:
    explicit SaveTo(QString path, int budgetMs = 5000) : answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &SaveTo::poll);
        timer.start();
    }
    int dialogs = 0;

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++dialogs;
            fd->setDirectory(QFileInfo(answer).absolutePath());
            fd->selectFile(answer);
            static_cast<QDialog *>(fd)->accept();
        }
    }
    QTimer timer;
    QString answer;
    int left;
};

// A flow past the circle fixture with a per-surf compute and a fix averaging
// it, run long enough for particles to actually strike the surface.
const char *const kValues = "fx fy fz etot";

class SurfReportLive : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        if (!*testLibrary() || fixtures().isEmpty()) return;
        sparta = new SpartaWrapper;
        if (!sparta->loadLib(testLibrary())) {
            delete sparta;
            sparta = nullptr;
            return;
        }
        char arg0[]  = "sparta";
        char *argv[] = {arg0, nullptr};
        sparta->open(1, argv);
        if (!sparta->isOpen()) {
            delete sparta;
            sparta = nullptr;
            return;
        }

        // the deck reads data.circle, air.species and air.vss by bare name
        runDir = new QTemporaryDir;
        for (const char *f : {"data.circle", "air.species", "air.vss"})
            QFile::copy(fixtures() + "/" + f, QDir(runDir->path()).filePath(f));

        const QString prev = QDir::currentPath();
        QDir::setCurrent(runDir->path());
        sparta->commandsString(deck());
        QDir::setCurrent(prev);
    }

    static void TearDownTestSuite()
    {
        if (sparta) sparta->close();
        delete sparta;
        sparta = nullptr;
        delete runDir;
        runDir = nullptr;
    }

    static QString deck()
    {
        return QStringList{"seed 12345",
                           "dimension 2",
                           "global gridcut 0.0 comm/sort yes",
                           "boundary o r p",
                           "create_box 0 10 0 10 -0.5 0.5",
                           "create_grid 10 10 1",
                           "balance_grid rcb cell",
                           "global nrho 1.0 fnum 0.001",
                           "species air.species N O",
                           "mixture air N O vstream 100.0 0 0",
                           "read_surf data.circle",
                           "surf_collide 1 diffuse 300.0 0.0",
                           "surf_modify all collide 1",
                           "collide vss air air.vss",
                           "timestep 0.0001",
                           "fix in emit/face air xlo twopass",
                           QString("compute 1 surf all air %1").arg(kValues),
                           "fix 1 ave/surf all 1 100 100 c_1[*]",
                           "stats 100",
                           "run 300"}
            .join('\n');
    }

    void SetUp() override
    {
        if (!sparta) GTEST_SKIP() << "no shared libsparta / no surface fixtures";
        ASSERT_EQ(sparta->extractSetting("surf_exist"), 1)
            << "the deck did not produce a surface to report on";
    }

    /// the dialog as the Run menu opens it
    SurfReportDialog *dialog() { return new SurfReportDialog(nullptr, sparta, deck()); }

    template <class W> static W *ctl(QWidget *w, const char *name)
    {
        return w->findChild<W *>(name);
    }

    static QString reportOf(SurfReportDialog *d)
    {
        return ctl<QPlainTextEdit>(d, "report")->toPlainText();
    }

    static void selectSource(SurfReportDialog *d, const QString &s)
    {
        auto *box = ctl<QComboBox>(d, "source");
        ASSERT_NE(box, nullptr);
        const int i = box->findText(s);
        ASSERT_GE(i, 0) << "the dialog does not offer " << s.toStdString() << "; it offers "
                        << [box] {
                               QStringList a;
                               for (int k = 0; k < box->count(); ++k) a << box->itemText(k);
                               return a.join(", ").toStdString();
                           }();
        box->setCurrentIndex(i);
    }

    static void compute(SurfReportDialog *d)
    {
        QMetaObject::invokeMethod(d, "computeReport");
    }

    /// the number printed after @p label in the report
    static double number(const QString &report, const QString &label)
    {
        const QRegularExpression re(QRegularExpression::escape(label) +
                                    R"(=\s*(-?[0-9.eE+-]+))");
        const auto m = re.match(report);
        EXPECT_TRUE(m.hasMatch()) << label.toStdString() << " is missing from:\n"
                                  << report.toStdString();
        return m.hasMatch() ? m.captured(1).toDouble() : std::nan("");
    }

    static SpartaWrapper *sparta;
    static QTemporaryDir *runDir;
    QTemporaryDir dir;
};

SpartaWrapper *SurfReportLive::sparta   = nullptr;
QTemporaryDir *SurfReportLive::runDir   = nullptr;

} // namespace

// ------------------------------------------------------------- what it offers

TEST_F(SurfReportLive, OffersTheComputesAndFixesTheRunActuallyDefined)
{
    auto *d = dialog();
    auto *box = ctl<QComboBox>(d, "source");
    ASSERT_NE(box, nullptr);
    QStringList offered;
    for (int i = 0; i < box->count(); ++i) offered << box->itemText(i);

    EXPECT_TRUE(offered.contains("c_1")) << offered.join(", ").toStdString();
    EXPECT_TRUE(offered.contains("f_1")) << offered.join(", ").toStdString();
    delete d;
}

TEST_F(SurfReportLive, LabelsTheColumnsFromTheDeckWithoutBeingTold)
{
    auto *d = dialog();
    selectSource(d, "c_1");
    EXPECT_EQ(ctl<QLineEdit>(d, "labels")->text(), QString(kValues).split(' ').join(", "))
        << "the compute's own value list was not recovered from the deck";
    delete d;
}

TEST_F(SurfReportLive, AFixThatAveragesAComputeInheritsThatComputesLabels)
{
    auto *d = dialog();
    selectSource(d, "f_1");
    EXPECT_EQ(ctl<QLineEdit>(d, "labels")->text(), QString(kValues).split(' ').join(", "))
        << "fix 1 averages c_1[*] and should be labelled with c_1's values";
    delete d;
}

// ------------------------------------------------------------- what it reads

TEST_F(SurfReportLive, ReadsOneRowPerSurfaceElementFromTheLiveCompute)
{
    auto *d = dialog();
    selectSource(d, "c_1");
    compute(d);
    const QString r = reportOf(d);

    const int nsurf = sparta->extractSetting("nlocal_surf") > 0
                          ? sparta->extractSetting("nlocal_surf")
                          : sparta->extractSetting("nsurf");
    ASSERT_GT(nsurf, 0);
    EXPECT_TRUE(r.contains(QString("Surface elements: %1").arg(nsurf)))
        << "the report did not read every element:\n"
        << r.toStdString();
    EXPECT_TRUE(r.contains("Integrated force")) << r.toStdString();
    delete d;
}

TEST_F(SurfReportLive, TheIntegratedForceIsTheSumOfTheElementsItExported)
{
    // closes the loop: the CSV is the array as read from SPARTA, so summing its
    // fx column has to reproduce the Fx the report printed.  A transposed read
    // or a stride mismatch breaks the two apart.
    auto *d = dialog();
    selectSource(d, "c_1");
    compute(d);
    const QString report = reportOf(d);

    const QString csv = dir.filePath("surf.csv");
    {
        SaveTo answer(csv);
        QMetaObject::invokeMethod(d, "exportCsv");
        QCoreApplication::processEvents();
        EXPECT_EQ(answer.dialogs, 1) << "Export CSV did not ask where to save";
    }
    ASSERT_TRUE(QFile::exists(csv)) << "nothing was exported";

    QFile f(csv);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    QTextStream ts(&f);
    const QStringList header = ts.readLine().split(',');
    const int fx = header.indexOf("fx"), fy = header.indexOf("fy");
    ASSERT_GE(fx, 0) << "the CSV header does not name the columns: " << header.join(",").toStdString();

    double sumx = 0.0, sumy = 0.0;
    int rows = 0, positive = 0, negative = 0;
    while (!ts.atEnd()) {
        const QStringList c = ts.readLine().split(',');
        if (c.size() <= fx) continue;
        const double v = c.at(fx).toDouble();
        sumx += v;
        sumy += c.at(fy).toDouble();
        if (v > 0) ++positive;
        if (v < 0) ++negative;
        ++rows;
    }

    // the sum check only means something over data that actually varies: an
    // all-zero array would satisfy it however badly the read went
    EXPECT_GT(positive, 0) << "no element recorded a positive fx";
    EXPECT_GT(negative, 0) << "every element recorded the same sign; the read may be constant";

    EXPECT_TRUE(report.contains(QString("Surface elements: %1").arg(rows)))
        << "the CSV has " << rows << " elements, the report says otherwise:\n"
        << report.toStdString();
    EXPECT_NEAR(number(report, "Fx"), sumx, std::abs(sumx) * 1e-5 + 1e-12)
        << "the integrated Fx is not the sum of the exported fx column";
    EXPECT_NEAR(number(report, "Fy"), sumy, std::abs(sumy) * 1e-5 + 1e-12)
        << "the integrated Fy is not the sum of the exported fy column";
    delete d;
}

TEST_F(SurfReportLive, TheReportedTotalMatchesAnIndependentReadOfTheSameArray)
{
    // The CSV check above closes a loop that both halves of walk the array the
    // same way, so a transposed read would agree with itself.  This walks it
    // separately, here in the test, and asks the report to match.
    const int nsurf = sparta->extractSetting("nlocal_surf") > 0
                          ? sparta->extractSetting("nlocal_surf")
                          : sparta->extractSetting("nsurf");
    ASSERT_GT(nsurf, 0);
    auto **a = static_cast<double **>(
        sparta->extractCompute("1", SpartaWrapper::SURF_STYLE, SpartaWrapper::ARRAY_TYPE));
    ASSERT_NE(a, nullptr) << "the per-surf compute did not return an array";

    const int ncol = QString(kValues).split(' ').size();
    double want[4] = {0, 0, 0, 0};
    for (int i = 0; i < nsurf; ++i)
        for (int j = 0; j < ncol && j < 4; ++j) want[j] += a[i][j];

    auto *d = dialog();
    selectSource(d, "c_1");
    compute(d);
    const QString r = reportOf(d);

    // the report prints six significant digits, so 1e-5 relative is as close as
    // a parsed value can be asked to come
    EXPECT_NEAR(number(r, "Fx"), want[0], std::abs(want[0]) * 1e-5 + 1e-30)
        << "the report's Fx is not the column the array actually holds:\n" << r.toStdString();
    EXPECT_NEAR(number(r, "Fy"), want[1], std::abs(want[1]) * 1e-5 + 1e-30)
        << "the report's Fy is not the column the array actually holds:\n" << r.toStdString();
    // etot is the fourth column, and reaches the report as the heat flux
    EXPECT_NEAR(number(r, "Q"), want[3], std::abs(want[3]) * 1e-5 + 1e-40)
        << "the report's heat flux is not the etot column:\n" << r.toStdString();
    delete d;
}

TEST_F(SurfReportLive, AReportOffersItsResultsForExportAndAnEmptyOneDoesNot)
{
    auto *d = dialog();
    auto *csv = ctl<QPushButton>(d, "csv");
    ASSERT_NE(csv, nullptr);
    EXPECT_FALSE(csv->isEnabled()) << "export was offered before anything had been computed";

    selectSource(d, "c_1");
    compute(d);
    EXPECT_TRUE(csv->isEnabled()) << "a completed report did not offer its data for export";
    delete d;
}

TEST_F(SurfReportLive, TheFixAndTheComputeItAveragesAgreeOnTheElementCount)
{
    // extractFix and extractCompute are different library calls over the same
    // surface; a style constant that is wrong for one of them shows up here
    auto *c = dialog();
    selectSource(c, "c_1");
    compute(c);
    auto *fx = dialog();
    selectSource(fx, "f_1");
    compute(fx);

    const QRegularExpression re(R"(Surface elements: (\d+))");
    const auto mc = re.match(reportOf(c)), mf = re.match(reportOf(fx));
    ASSERT_TRUE(mc.hasMatch()) << reportOf(c).toStdString();
    ASSERT_TRUE(mf.hasMatch()) << reportOf(fx).toStdString();
    EXPECT_EQ(mc.captured(1), mf.captured(1));
    delete c;
    delete fx;
}

TEST_F(SurfReportLive, TheReportNamesTheTimestepItWasTakenAt)
{
    auto *d = dialog();
    selectSource(d, "c_1");
    compute(d);
    EXPECT_TRUE(reportOf(d).contains("at timestep 300"))
        << "the report does not say when it was taken:\n"
        << reportOf(d).toStdString();
    delete d;
}

// ------------------------------------------------------------- refusals

TEST_F(SurfReportLive, ASourceThatIsNotPerSurfaceIsRefusedRatherThanMisread)
{
    // "fix in" is an emit/face fix with no per-surf array at all.  Reading one
    // anyway would hand the reduction whatever happened to be at that address.
    auto *d = dialog();
    auto *box = ctl<QComboBox>(d, "source");
    const int i = box->findText("f_in");
    if (i < 0) GTEST_SKIP() << "the emit fix is not offered";
    box->setCurrentIndex(i);
    ctl<QLineEdit>(d, "labels")->setText("fx, fy"); // it has none to derive
    compute(d);

    const QString r = reportOf(d);
    EXPECT_TRUE(r.contains("did not return a per-surface array") ||
                r.contains("not a readable per-surface"))
        << "a non-surface fix was reported on anyway:\n"
        << r.toStdString();
    EXPECT_FALSE(ctl<QPushButton>(d, "csv")->isEnabled())
        << "a refused source still offered its results for export";
    delete d;
}

TEST_F(SurfReportLive, WithoutLabelsItAsksForThemRatherThanGuessing)
{
    auto *d = dialog();
    selectSource(d, "c_1");
    ctl<QLineEdit>(d, "labels")->clear();
    compute(d);
    EXPECT_TRUE(reportOf(d).contains("Enter the value labels")) << reportOf(d).toStdString();
    delete d;
}

TEST_F(SurfReportLive, ExportBeforeAnyReportWritesNothing)
{
    auto *d = dialog();
    const QString csv = dir.filePath("empty.csv");
    SaveTo answer(csv);
    QMetaObject::invokeMethod(d, "exportCsv");
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.dialogs, 0) << "an empty report still asked where to save";
    EXPECT_FALSE(QFile::exists(csv));
    delete d;
}

TEST_F(SurfReportLive, AComputeClearedBySetupSaysSoInsteadOfPrintingZeros)
{
    // Creating an image renders through `run 0 pre yes post no`, and that setup
    // discards a `compute surf`'s accumulated tallies.  The report then reads
    // back all zeros, which is indistinguishable from a surface nothing ever
    // hit -- so it has to say which it is.  `run 0` here is exactly what the
    // render does, without needing an image.
    //
    // The instance is shared by the whole fixture, so the tallies are put back
    // before returning: this test must leave the next one the data it expects,
    // whatever order they run in.
    auto *d = dialog();
    selectSource(d, "c_1");

    sparta->commandsString("run 0 pre yes post no");
    compute(d);
    const QString cleared = reportOf(d);
    EXPECT_TRUE(cleared.contains("read back as all zeros"))
        << "a report with nothing in it was presented as a result:\n"
        << cleared.toStdString();
    EXPECT_TRUE(cleared.contains("fix ave/surf"))
        << "the note does not say what to do about it:\n"
        << cleared.toStdString();

    sparta->commandsString("run 100");
    compute(d);
    const QString restored = reportOf(d);
    EXPECT_FALSE(restored.contains("read back as all zeros"))
        << "the note is printed over real data as well, so it says nothing:\n"
        << restored.toStdString();
    delete d;
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
