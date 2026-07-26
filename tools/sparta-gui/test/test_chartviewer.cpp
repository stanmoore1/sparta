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

// The chart window: the panel that turns thermo output into plots, and the
// standalone plot window the same class becomes when handed a data file
// instead of a running simulation.
//
// It is the second-largest source in the GUI and was the largest wholly
// untested one.  Nothing here needs a simulator: ChartWindow takes a null
// SpartaGui, which is exactly the standalone-plot mode a user gets from
// "File > Plot data file", and every method below is reachable from there.

#include "chartviewer.h"

#include "plotdata.h"
#include "plotwidget.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QSignalSpy>
#include <QAbstractSpinBox>
#include <QSpinBox>
#include <QTemporaryDir>
#include <QTest>

#include <cmath>

namespace {

// A three-column table: a step column and two data columns with distinct
// shapes, so a chart bound to the wrong column is visible in the bounds.
PlotData table(int rows = 40)
{
    PlotData d;
    d.setColumnNames({"Step", "Temp", "Press"});
    for (int i = 0; i < rows; ++i)
        d.appendRow({static_cast<double>(i * 10), 300.0 + i, 1.0e5 - 100.0 * i});
    return d;
}

// The chart window as "File > Plot data file" builds it: no main window behind
// it, so no simulation and no live thermo feed.
class Chart : public ::testing::Test {
protected:
    void SetUp() override
    {
        win = new ChartWindow(QString(), nullptr);
        win->resize(800, 600);
    }
    void TearDown() override { delete win; }

    // by object name, not by position: findChildren<QLineEdit*> also returns
    // the editor every QSpinBox embeds, and in standalone mode the X-axis field
    // sits between the title and the Y-axis field, so positional lookup here
    // silently addresses the wrong control -- convincingly enough that it
    // accepts setText() and emits editingFinished() like the real one
    template <class W> W *ctl(const char *name) const
    {
        auto *w = win->findChild<W *>(QLatin1String(name));
        EXPECT_NE(w, nullptr) << "no control named " << name;
        return w;
    }

    // the controls have no object names, so reach them by type and position --
    // which is itself worth asserting, since it pins the layout down
    template <class W> QList<W *> all() const { return win->findChildren<W *>(); }


    ChartWindow *win = nullptr;
};

} // namespace

// ---------------------------------------------------------------- construction

TEST_F(Chart, StartsEmpty)
{
    EXPECT_EQ(win->numCharts(), 0);
    EXPECT_EQ(win->getStep(), -1) << "an empty window has no last step to report";
}

TEST_F(Chart, BuildsItsControls)
{
    // one combo per chart selector and one per smoothing choice, two spin boxes
    // for the Savitzky-Golay window and order, and the label editors
    EXPECT_GE(all<QComboBox>().size(), 2);
    EXPECT_GE(all<QSpinBox>().size(), 2);
    EXPECT_NE(ctl<QLineEdit>("chartTitle"), nullptr);
    EXPECT_NE(ctl<QLineEdit>("chartYlabel"), nullptr);
    EXPECT_NE(ctl<QLineEdit>("chartXlabel"), nullptr) << "standalone plot mode has an X-axis label";
    EXPECT_GE(all<QCheckBox>().size(), 1);
}

// ---------------------------------------------------------------- the live path

TEST_F(Chart, AddingAChartMakesItActiveAndSelectable)
{
    win->addChart("Temp", 3);
    EXPECT_EQ(win->numCharts(), 1);
    EXPECT_TRUE(win->hasTitle("Temp", 0));

    win->addChart("Press", 4);
    EXPECT_EQ(win->numCharts(), 2);
    EXPECT_TRUE(win->hasTitle("Press", 1));
    EXPECT_TRUE(win->hasTitle("Temp", 0)) << "the second chart displaced the first";
}

TEST_F(Chart, DataGoesToTheChartWithTheMatchingThermoColumn)
{
    win->addChart("Temp", 3);
    win->addChart("Press", 4);

    for (int i = 0; i < 5; ++i) {
        win->addData(i * 100, 300.0 + i, 3);
        win->addData(i * 100, 1.0e5, 4);
    }
    EXPECT_EQ(win->getStep(), 400) << "getStep reports the last x of the first chart";
}

TEST_F(Chart, DataForAnUnknownColumnIsDropped)
{
    win->addChart("Temp", 3);
    win->addData(100, 300.0, 3);
    win->addData(200, 999.0, 77); // no such thermo column
    EXPECT_EQ(win->getStep(), 100) << "the unmatched point was appended anyway";
}

TEST_F(Chart, ResettingClearsEveryChart)
{
    win->addChart("Temp", 3);
    win->addChart("Press", 4);
    win->addData(100, 300.0, 3);

    win->resetCharts();
    EXPECT_EQ(win->numCharts(), 0);
    EXPECT_EQ(win->getStep(), -1);

    // and the window is reusable rather than merely emptied
    win->addChart("Temp", 3);
    win->addData(500, 42.0, 3);
    EXPECT_EQ(win->numCharts(), 1);
    EXPECT_EQ(win->getStep(), 500);
}

TEST_F(Chart, ResetZoomOnAnEmptyWindowIsHarmless)
{
    win->resetZoom(); // no charts: must not reach through a null view
    win->addChart("Temp", 3);
    win->addData(0, 1.0, 3);
    win->resetZoom();
    SUCCEED();
}

// ---------------------------------------------------------------- the file path

TEST_F(Chart, LoadingAFileBuildsOneChartPerSelectedColumn)
{
    win->loadData(table(), 0, {1, 2});
    EXPECT_EQ(win->numCharts(), 2);
    EXPECT_TRUE(win->hasTitle("Temp", 0));
    EXPECT_TRUE(win->hasTitle("Press", 1));
    EXPECT_EQ(win->getStep(), 390) << "the last step of a 40-row table stepping by 10";
}

TEST_F(Chart, LoadingReplacesWhatWasThereBefore)
{
    win->loadData(table(), 0, {1, 2});
    win->loadData(table(), 0, {1});
    EXPECT_EQ(win->numCharts(), 1) << "the previous two charts survived the reload";
    EXPECT_TRUE(win->hasTitle("Temp", 0));
}

TEST_F(Chart, LoadingRefusesTheDegenerateCases)
{
    win->loadData(PlotData{}, 0, {1});
    EXPECT_EQ(win->numCharts(), 0) << "an empty table";

    win->loadData(table(), 0, {});
    EXPECT_EQ(win->numCharts(), 0) << "no y columns selected";

    win->loadData(table(), -1, {1});
    EXPECT_EQ(win->numCharts(), 0) << "a negative x column";

    win->loadData(table(), 99, {1});
    EXPECT_EQ(win->numCharts(), 0) << "an x column past the end";
}

TEST_F(Chart, AnOutOfRangeYColumnIsSkippedRatherThanPlotted)
{
    win->loadData(table(), 0, {1, 99, 2});
    EXPECT_EQ(win->numCharts(), 2) << "the bogus column produced a chart";
    EXPECT_TRUE(win->hasTitle("Temp", 0));
    EXPECT_TRUE(win->hasTitle("Press", 1)) << "the skip left a hole in the numbering";
}

TEST_F(Chart, ASingleRowTableStillPlots)
{
    win->loadData(table(1), 0, {1});
    EXPECT_EQ(win->numCharts(), 1);
    EXPECT_EQ(win->getStep(), 0);
    win->resetZoom(); // a zero-width range is where an axis autoscale divides by zero
    SUCCEED();
}

// ---------------------------------------------------------------- switching charts

TEST_F(Chart, SwitchingChartsRestoresThatChartsYLabel)
{
    win->loadData(table(), 0, {1, 2});
    auto *columns = ctl<QComboBox>("columns");
    ASSERT_EQ(columns->count(), 2);

    // rename chart 0's y axis, switch away and back
    auto *ylabel = ctl<QLineEdit>("chartYlabel");
    const QString before = ylabel->text();

    columns->setCurrentIndex(1);
    QCoreApplication::processEvents();
    columns->setCurrentIndex(0);
    QCoreApplication::processEvents();

    EXPECT_EQ(ylabel->text(), before) << "chart 0 came back with another chart's label";
}

TEST_F(Chart, TheChartSelectorDrivesTheView)
{
    win->loadData(table(), 0, {1, 2});
    auto *columns = ctl<QComboBox>("columns");
    for (int i = 0; i < columns->count(); ++i) {
        columns->setCurrentIndex(i);
        QCoreApplication::processEvents();
    }
    EXPECT_EQ(win->numCharts(), 2) << "cycling the selector disturbed the charts";
}

// ---------------------------------------------------------------- smoothing

TEST_F(Chart, EverySmoothingChoiceIsSurvivable)
{
    win->loadData(table(), 0, {1});
    auto *smooth = ctl<QComboBox>("smooth");
    ASSERT_GE(smooth->count(), 2);

    for (int i = 0; i < smooth->count(); ++i) {
        smooth->setCurrentIndex(i);
        QCoreApplication::processEvents();
        EXPECT_EQ(win->numCharts(), 1) << "choice " << i << " lost the chart";
    }
}

TEST_F(Chart, TheSavitzkyGolayParametersOnlyApplyWhileSmoothing)
{
    win->loadData(table(), 0, {1});
    auto *smooth = ctl<QComboBox>("smooth");
    auto *window = ctl<QSpinBox>("smoothWindow");
    auto *order  = ctl<QSpinBox>("smoothOrder");

    smooth->setCurrentIndex(0); // raw only
    QCoreApplication::processEvents();
    const bool rawEnabled = window->isEnabled();

    smooth->setCurrentIndex(1); // the processed slot
    QCoreApplication::processEvents();
    EXPECT_TRUE(window->isEnabled()) << "the smoothing window is dead while smoothing";
    EXPECT_TRUE(order->isEnabled());
    EXPECT_FALSE(rawEnabled) << "the smoothing window is live with nothing to smooth";
}

TEST_F(Chart, ChangingTheSmoothingParametersRedraws)
{
    win->loadData(table(), 0, {1});
    auto *smooth = ctl<QComboBox>("smooth");
    smooth->setCurrentIndex(1);
    QCoreApplication::processEvents();

    auto *window = ctl<QSpinBox>("smoothWindow");
    auto *order  = ctl<QSpinBox>("smoothOrder");
    window->setValue(window->value() + 2);
    order->setValue(std::min(order->value() + 1, order->maximum()));
    QCoreApplication::processEvents();
    EXPECT_EQ(win->numCharts(), 1);
}

TEST_F(Chart, SmoothingAppliesToEveryChartNotJustTheVisibleOne)
{
    win->loadData(table(), 0, {1, 2});
    auto *smooth = ctl<QComboBox>("smooth");
    smooth->setCurrentIndex(1);
    QCoreApplication::processEvents();

    // switch to the chart that was hidden when smoothing was turned on: if the
    // flags had only been set on the active column, this would show raw data
    auto *columns = ctl<QComboBox>("columns");
    columns->setCurrentIndex(1);
    QCoreApplication::processEvents();
    EXPECT_EQ(win->numCharts(), 2);
}

// ---------------------------------------------------------------- labels

TEST_F(Chart, EditingTheLabelsIsAccepted)
{
    win->loadData(table(), 0, {1});
    for (auto *e : {ctl<QLineEdit>("chartTitle"), ctl<QLineEdit>("chartYlabel"),
                    ctl<QLineEdit>("chartXlabel")}) {
        e->setText("relabelled");
        emit e->editingFinished();
        QCoreApplication::processEvents();
    }
    EXPECT_EQ(win->numCharts(), 1);
}

TEST_F(Chart, TheYLabelIsPerChartAndTheTitleIsNot)
{
    win->loadData(table(), 0, {1, 2});
    auto *columns = ctl<QComboBox>("columns");
    auto *title  = ctl<QLineEdit>("chartTitle");
    auto *ylabel = ctl<QLineEdit>("chartYlabel");

    title->setText("shared title");
    emit title->editingFinished();
    ylabel->setText("chart zero");
    emit ylabel->editingFinished();
    QCoreApplication::processEvents();

    columns->setCurrentIndex(1);
    QCoreApplication::processEvents();
    EXPECT_EQ(title->text(), "shared title") << "the title followed the chart switch";
    EXPECT_NE(ylabel->text(), "chart zero") << "chart 1 inherited chart 0's y label";

    columns->setCurrentIndex(0);
    QCoreApplication::processEvents();
    EXPECT_EQ(ylabel->text(), "chart zero") << "chart 0 lost its own y label";
}

TEST_F(Chart, LabelEditsOnAnEmptyWindowAreHarmless)
{
    for (auto *e : {ctl<QLineEdit>("chartTitle"), ctl<QLineEdit>("chartYlabel"),
                    ctl<QLineEdit>("chartXlabel")}) {
        e->setText("x");
        emit e->editingFinished();
    }
    QCoreApplication::processEvents();
    SUCCEED(); // no charts: every label slot must return early rather than index -1
}

// ---------------------------------------------------------------- ranges

TEST_F(Chart, TheRangeSlidersNarrowTheView)
{
    win->loadData(table(), 0, {1});
    win->setRangeEnabled(true);
    // private slots: reached the way the range sliders reach them
    QMetaObject::invokeMethod(win, "updateXRange", Q_ARG(int, 10), Q_ARG(int, 90));
    QMetaObject::invokeMethod(win, "updateYRange", Q_ARG(int, 90), Q_ARG(int, 10));
    win->resetZoom();
    EXPECT_EQ(win->numCharts(), 1);
}

TEST_F(Chart, RangeUpdatesOnAnEmptyWindowAreHarmless)
{
    QMetaObject::invokeMethod(win, "updateXRange", Q_ARG(int, 0), Q_ARG(int, 100));
    QMetaObject::invokeMethod(win, "updateYRange", Q_ARG(int, 0), Q_ARG(int, 100));
    SUCCEED();
}

TEST_F(Chart, RangeControlsCanBeDisabledWholesale)
{
    win->loadData(table(), 0, {1});
    win->setRangeEnabled(false);
    auto *smooth = ctl<QComboBox>("smooth");
    EXPECT_FALSE(smooth->isEnabled());
    win->setRangeEnabled(true);
    EXPECT_TRUE(smooth->isEnabled());
}

// ---------------------------------------------------------------- units and normalisation

TEST_F(Chart, UnitsAndNormalisationAreSettable)
{
    win->setUnits("si");
    win->setNorm(true);
    auto *norm = ctl<QCheckBox>("norm");
    EXPECT_TRUE(norm->isChecked());
    win->setNorm(false);
    EXPECT_FALSE(norm->isChecked());
}

// ---------------------------------------------------------------- the view itself

TEST_F(Chart, TheViewReportsTheBoundsOfWhatItHolds)
{
    win->loadData(table(), 0, {1});
    auto *view = ctl<ChartViewer>("chartView");
    ASSERT_NE(view, nullptr);

    const QRectF box = view->getMinMax();
    EXPECT_DOUBLE_EQ(box.left(), 0.0);
    EXPECT_DOUBLE_EQ(box.right(), 390.0);
    EXPECT_EQ(view->getCount(), 40);
}

TEST_F(Chart, TheViewSurvivesBeingUnbound)
{
    win->loadData(table(), 0, {1});
    auto *view = ctl<ChartViewer>("chartView");
    ASSERT_NE(view, nullptr);
    view->setColumn(nullptr); // the state resetCharts() leaves the view in
    view->resetZoom();
    view->updateSmooth();
    EXPECT_TRUE(view->getMinMax().isNull()) << "an unbound view invented bounds";
}

TEST_F(Chart, ExplicitAxisRangesStick)
{
    win->loadData(table(), 0, {1});
    auto *view = ctl<ChartViewer>("chartView");
    ASSERT_NE(view, nullptr);
    view->setXAxisRange(100.0, 200.0);
    view->setYAxisRange(310.0, 320.0);
    view->resetZoom(); // and back to the data bounds
    const QRectF box = view->getMinMax();
    EXPECT_DOUBLE_EQ(box.right(), 390.0);
}

// ---------------------------------------------------------------- painting

TEST_F(Chart, ItRendersWithoutADisplay)
{
    win->loadData(table(), 0, {1, 2});
    auto *view = ctl<ChartViewer>("chartView");
    ASSERT_NE(view, nullptr);

    const QPixmap shot = view->grab();
    ASSERT_FALSE(shot.isNull());
    EXPECT_GT(shot.width(), 0);

    // the same grab the Copy and Save As actions take, so a crash in paint()
    // would fail here rather than in a file dialog nobody can open
    const QImage img = shot.toImage();
    bool anyInk      = false;
    const QRgb first = img.pixel(0, 0);
    for (int y = 0; y < img.height() && !anyInk; y += 4)
        for (int x = 0; x < img.width(); x += 4)
            if (img.pixel(x, y) != first) { anyInk = true; break; }
    EXPECT_TRUE(anyInk) << "the chart rendered as a flat field of one colour";
}

TEST_F(Chart, AnEmptyChartRendersToo)
{
    auto *view = ctl<ChartViewer>("chartView");
    ASSERT_NE(view, nullptr);
    EXPECT_FALSE(view->grab().isNull());
}

// ---------------------------------------------------------------- teardown

TEST_F(Chart, ClosingWithChartsLoadedIsClean)
{
    win->loadData(table(), 0, {1, 2});
    win->close();
    QCoreApplication::processEvents();
    SUCCEED();
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    // a private settings scope: the window reads chart defaults from QSettings,
    // and parallel ctest processes must not share (or clobber) one another's
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_chartviewer.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
