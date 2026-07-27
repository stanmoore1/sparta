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

// Plotting a data file: SpartaGui::plotDataFile() and ChartWindow::addDataFile().
//
// These are how a user gets somebody else's numbers onto a chart -- an
// experimental reference curve, a previous run's output, a table from a paper.
// The parsers underneath are covered (test_plotdata.cpp) and so is the column
// picker (test_plotdatadialog.cpp), but neither end had ever been driven: both
// of these sit behind a file dialog and then a second modal dialog, so nothing
// had ever checked that the columns the user picked are the ones that reach the
// chart.
//
// That is worth checking rather than assuming, because the failure is silent.
// A curve plotted from the wrong column is still a curve, and the axis labels
// come from the same picker, so it will even be labelled convincingly.

#include "spartagui.h"

#include "chartviewer.h"
#include "constants.h"
#include "chartdialogs.h"
#include "plotdatadialog.h"

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QIcon>
#include <QHash>
#include <QLineEdit>
#include <QMessageBox>
#include <QRadioButton>
#include <QSettings>
#include <QTemporaryDir>
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

/// Drives the two modals this path puts up in order: the file dialog, then the
/// column picker.  @p xcol and @p ycols, when given, are set on the picker
/// before it is accepted, so a test can plot a column other than the default.
class Answer : public QObject {
public:
    explicit Answer(QString path, bool acceptPicker = true, int budgetMs = 10000) :
        path(std::move(path)), acceptPicker(acceptPicker), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answer::poll);
        timer.start();
    }

    int xcol = -1;               ///< which column to use as x (-1 = leave the default)
    QList<int> ycols;            ///< which columns to plot (empty = leave the defaults)
    QStringList renames;         ///< per-column name overrides, by index
    int fileDialogs = 0;
    int pickers     = 0;
    QStringList messages;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;

        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->text() + " " + box->informativeText();
            box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++fileDialogs;
            if (path.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(path).absolutePath());
                fd->selectFile(path);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *pd = qobject_cast<PlotDataDialog *>(m)) {
            ++pickers;
            if (!acceptPicker) {
                pd->reject();
                return;
            }
            // the picker lays out one radio (x), one check (y) and one name
            // field per column, in column order
            const auto radios = pd->findChildren<QRadioButton *>();
            const auto checks = pd->findChildren<QCheckBox *>();
            const auto names  = pd->findChildren<QLineEdit *>();
            if (xcol >= 0 && xcol < radios.size()) radios.at(xcol)->setChecked(true);
            if (!ycols.isEmpty())
                for (int i = 0; i < checks.size(); ++i) checks.at(i)->setChecked(ycols.contains(i));
            for (int i = 0; i < renames.size() && i < names.size(); ++i)
                if (!renames.at(i).isEmpty()) names.at(i)->setText(renames.at(i));
            pd->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }

    QTimer timer;
    QString path;
    bool acceptPicker;
    int left;
};

class DataFile : public ::testing::Test {
protected:
    void SetUp() override
    {
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.setValue(Keys::RESTORE_SESSION, false);
        settings.sync();
        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());
        closeCharts();
    }

    void TearDown() override
    {
        closeCharts();
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir);
        QSettings().clear();
    }

    static void closeCharts()
    {
        for (auto *w : QApplication::topLevelWidgets())
            if (auto *c = qobject_cast<ChartWindow *>(w)) c->close(); // WA_DeleteOnClose
        QApplication::processEvents();
    }

    static ChartWindow *newestChart()
    {
        ChartWindow *found = nullptr;
        for (auto *w : QApplication::topLevelWidgets())
            if (auto *c = qobject_cast<ChartWindow *>(w)) found = c;
        return found;
    }

    /// A three-column table whose values are separable on sight: x is 1..5,
    /// "twice" is 2x and "ten" is 10x, so which column was plotted is readable
    /// off the y-range alone.
    QString csv(const QString &name = "data.csv") const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write("x,twice,ten\n");
        for (int i = 1; i <= 5; ++i)
            f.write(QString("%1,%2,%3\n").arg(i).arg(2 * i).arg(10 * i).toUtf8());
        f.close();
        return p;
    }

    /// A two-column file: x 1..5 and one y column scaled by @p k, so each
    /// overlay of it adds exactly one series.
    QString oneColumnCsv(const QString &name, int k) const
    {
        QString t = "x,y\n";
        for (int i = 1; i <= 5; ++i) t += QString("%1,%2\n").arg(i).arg(k * i);
        return write(name, t);
    }

    QString write(const QString &name, const QString &text) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
        f.close();
        return p;
    }

    SpartaGui *window()
    {
        if (!gui) gui = new SpartaGui(nullptr, QString(), 800, 600);
        return gui;
    }

    /// Two reapers must never be alive together: they both answer the next
    /// modal, and whichever polls first wins.  Every call here scopes its own.
    void plotFile(const QString &path)
    {
        Answer answer(path);
        QMetaObject::invokeMethod(window(), "plotDataFile");
        QApplication::processEvents();
    }

    void overlayFile(ChartWindow *chart, const QString &path)
    {
        Answer answer(path);
        QMetaObject::invokeMethod(chart, "addDataFile");
        QApplication::processEvents();
    }

    /// The data range of the chart on display, with the chart's own y padding
    /// removed so the numbers can be compared with the file's.
    struct Range {
        double xlo, xhi, ylo, yhi;
    };
    static Range rangeOf(ChartWindow *w)
    {
        auto *v = w->findChild<ChartViewer *>();
        EXPECT_NE(v, nullptr);
        if (!v) return {0, 0, 0, 0};
        const QRectF b = v->getMinMax();
        const double lo = qMin(b.top(), b.bottom()), hi = qMax(b.top(), b.bottom());
        const double pad =
            Cfg::CHART_YPAD_FRACTION * (hi - lo) / (1 + 2 * Cfg::CHART_YPAD_FRACTION);
        return {b.left(), b.right(), lo + pad, hi - pad};
    }

    QTemporaryDir dir;
    QString startDir;
    SpartaGui *gui = nullptr;
};

} // namespace

// ------------------------------------------------------- plotting a new file

TEST_F(DataFile, PlotsTheFileItWasGivenWithTheColumnsItWasShown)
{
    Answer answer(csv());
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "it did not ask which file";
    EXPECT_EQ(answer.pickers, 1) << "it did not ask which columns";

    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr) << "no chart window opened: " << answer.all().toStdString();
    EXPECT_TRUE(chart->windowTitle().contains("data.csv"))
        << "the chart does not say what it is plotting: " << chart->windowTitle().toStdString();

    // both non-x columns become charts, and the window shows one at a time
    EXPECT_EQ(chart->numCharts(), 2) << "the default selection did not plot every other column";

    // x is the first column (1..5); the chart on display is the first y column,
    // "twice", which runs 2..10
    const Range r = rangeOf(chart);
    EXPECT_DOUBLE_EQ(r.xlo, 1.0);
    EXPECT_DOUBLE_EQ(r.xhi, 5.0);
    EXPECT_NEAR(r.ylo, 2.0, 1e-9) << "the smallest plotted value is not the file's";
    EXPECT_NEAR(r.yhi, 10.0, 1e-9) << "the largest plotted value is not the file's";
}

TEST_F(DataFile, PlotsTheColumnThatWasSelectedRatherThanTheFirstOne)
{
    // the check the picker exists for: a curve from the wrong column is still a
    // curve, and it will be labelled convincingly
    Answer answer(csv());
    answer.ycols = {1}; // "twice" only, so the y range is 2..10 and not 10..50
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    const Range r = rangeOf(chart);
    EXPECT_NEAR(r.yhi, 10.0, 1e-9)
        << "a column other than the selected one was plotted (y reached " << r.yhi << ")";
    EXPECT_NEAR(r.ylo, 2.0, 1e-9);
}

TEST_F(DataFile, TheChosenXColumnIsTheAbscissa)
{
    Answer answer(csv());
    answer.xcol  = 2; // "ten" on the x axis: 10..50
    answer.ycols = {0};
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    const Range r = rangeOf(chart);
    EXPECT_DOUBLE_EQ(r.xlo, 10.0) << "the x axis is not the column that was picked";
    EXPECT_DOUBLE_EQ(r.xhi, 50.0);
    EXPECT_NEAR(r.yhi, 5.0, 1e-9);
}

TEST_F(DataFile, CancellingTheFileDialogPlotsNothing)
{
    Answer answer{QString()}; // cancel
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_EQ(answer.pickers, 0) << "it went on to ask about columns of a file it never opened";
    EXPECT_EQ(newestChart(), nullptr);
}

TEST_F(DataFile, CancellingTheColumnPickerPlotsNothing)
{
    Answer answer(csv(), /*acceptPicker=*/false);
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    EXPECT_EQ(answer.pickers, 1);
    EXPECT_EQ(newestChart(), nullptr) << "cancelling the picker still opened a chart";
}

TEST_F(DataFile, SelectingNoColumnsSaysSoRatherThanOpeningAnEmptyChart)
{
    Answer answer(csv());
    answer.ycols = {}; // an explicit empty selection
    answer.ycols.clear();
    // uncheck everything by selecting a set that contains no valid index
    answer.ycols = QList<int>{99};
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    EXPECT_TRUE(answer.said("No data columns")) << answer.all().toStdString();
    EXPECT_EQ(newestChart(), nullptr) << "an empty chart was opened anyway";
}

TEST_F(DataFile, AFileThatIsNotDataIsRefusedWithItsReason)
{
    Answer answer(write("garbage.csv", "this is not a table\nnor is this\n"));
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();

    EXPECT_TRUE(answer.said("Could not read data")) << answer.all().toStdString();
    EXPECT_EQ(answer.pickers, 0) << "it asked which columns of a file it could not read";
    EXPECT_EQ(newestChart(), nullptr);
}

TEST_F(DataFile, SomethingThatIsNotAFileIsRefused)
{
    // a directory: an open dialog will hand one over, and it cannot be read as
    // a table.  (A path that does not exist is not reachable this way -- the
    // dialog refuses to accept one -- so there is nothing to test there.)
    // a short budget: an open dialog descends into a directory rather than
    // accepting it, so the reaper gives up rather than spinning for its default
    QDir(dir.path()).mkdir("adirectory.csv");
    Answer answer(dir.filePath("adirectory.csv"), true, 1500);
    QMetaObject::invokeMethod(window(), "plotDataFile");
    QApplication::processEvents();
    EXPECT_EQ(newestChart(), nullptr) << "a directory was plotted as data";
}

// ------------------------------------------- adding a file to an open chart

TEST_F(DataFile, AddsASecondFileAsAnOverlayOnTheChartAlreadyShown)
{
    // the reference-curve case: the existing chart keeps its own data and the
    // new file is drawn over it
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);
    const int before = view->overlaySeriesCount();

    // a second file whose values sit well above the first, so the combined
    // range shows it was actually added rather than quietly dropped
    QString p = dir.filePath("ref.csv");
    QFile f(p);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
    f.write("x,high\n");
    for (int i = 1; i <= 5; ++i) f.write(QString("%1,%2\n").arg(i).arg(100 * i).toUtf8());
    f.close();

    overlayFile(chart, p);

    EXPECT_GT(view->overlaySeriesCount(), before) << "the second file added no series";
    // the overlay is included in the chart's own range, so the axis now has to
    // reach the reference curve: added but not drawn would leave it at 10
    EXPECT_NEAR(rangeOf(chart).yhi, 500.0, 1e-9)
        << "the overlay is not on the chart's scale; it was added but not drawn";
}

TEST_F(DataFile, AddingToAChartWithNothingInItDoesNothing)
{
    ChartWindow w("empty", nullptr);
    Answer answer(csv());
    QMetaObject::invokeMethod(&w, "addDataFile");
    QApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 0)
        << "it asked for a file to overlay on a chart that has no data";
}

TEST_F(DataFile, CancellingTheOverlayLeavesTheChartAsItWas)
{
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);
    const int before = view->overlaySeriesCount();
    const double yhi = rangeOf(chart).yhi;

    overlayFile(chart, QString()); // cancel the file dialog

    EXPECT_EQ(view->overlaySeriesCount(), before);
    EXPECT_DOUBLE_EQ(rangeOf(chart).yhi, yhi) << "cancelling still changed the chart";
}

TEST_F(DataFile, AnUnreadableOverlayFileIsReportedAndAddsNothing)
{
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    auto *view = chart->findChild<ChartViewer *>();
    const int before = view->overlaySeriesCount();

    Answer second(write("junk.csv", "not a table at all\n"));
    QMetaObject::invokeMethod(chart, "addDataFile");
    QApplication::processEvents();
    EXPECT_TRUE(second.said("Could not read data")) << second.all().toStdString();
    EXPECT_EQ(view->overlaySeriesCount(), before);
}

TEST_F(DataFile, EachOverlayGetsItsOwnColour)
{
    // two overlays in the same colour are indistinguishable on the chart, which
    // defeats the point of adding a second reference curve
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);

    for (int k = 0; k < 2; ++k) overlayFile(chart, csv(QString("ref%1.csv").arg(k)));
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);
    EXPECT_GE(view->overlaySeriesCount(), 4) << "two files of two columns each were not all added";
    EXPECT_NE(overlaySeriesColor(0), overlaySeriesColor(1)) << "consecutive overlays share a colour";
}

TEST_F(DataFile, CancellingTheColumnPickerAddsNoOverlay)
{
    // the file dialog and the picker are two separate refusals; cancelling the
    // second one still has a file in hand and must not use it
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);
    const int before = view->overlaySeriesCount();

    Answer answer(oneColumnCsv("ref.csv", 100), /*acceptPicker=*/false);
    QMetaObject::invokeMethod(chart, "addDataFile");
    QApplication::processEvents();

    EXPECT_EQ(answer.pickers, 1) << "it never got as far as asking which columns";
    EXPECT_EQ(view->overlaySeriesCount(), before)
        << "cancelling the column picker added the overlay anyway";
}

TEST_F(DataFile, TwoOverlaysAreDrawnInDifferentColours)
{
    // two reference curves in one colour are indistinguishable on the chart,
    // which defeats the point of adding the second.  There is no accessor for a
    // series colour, so this asks the pixels: each series contributes its own
    // strong colour to the render.
    plotFile(csv("run.csv"));
    auto *chart = newestChart();
    ASSERT_NE(chart, nullptr);
    auto *view = chart->findChild<ChartViewer *>();
    ASSERT_NE(view, nullptr);

    overlayFile(chart, oneColumnCsv("refa.csv", 2));
    overlayFile(chart, oneColumnCsv("refb.csv", 3));
    ASSERT_EQ(view->overlaySeriesCount(), 2) << "the two overlays were not both added";

    chart->resize(600, 400);
    chart->show();
    QApplication::processEvents();
    const QImage img = view->grab().toImage();
    ASSERT_FALSE(img.isNull());

    // colours that are actually inked in quantity, ignoring the greys of the
    // background, the grid and the axes
    QHash<QRgb, int> hits;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) {
            const QColor c = img.pixelColor(x, y);
            if (c.saturation() > 100 && c.value() > 60) ++hits[c.rgb()];
        }
    int strong = 0;
    for (auto it = hits.constBegin(); it != hits.constEnd(); ++it)
        if (it.value() >= 20) ++strong;

    EXPECT_GE(strong, 3) << "the chart's own curve and its two overlays are not three "
                            "distinguishable colours; only " << strong << " were drawn";
    chart->hide();
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
        QString("test_datafileplot-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
