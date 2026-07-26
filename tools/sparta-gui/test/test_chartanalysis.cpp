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

// What ChartWindow::postProcess() does after its dialog is answered: seven
// analyses, each of which reads the chart's data, computes something, and puts
// the answer back on the chart as a fit curve or a reference line -- then
// reports it in a message box.
//
// Splitting the dialog out (chartdialogs.cpp) left this half still needing a
// chart with data in it, which is why it stayed uncovered.  It does not,
// however, need a simulator: a ChartWindow loaded from a PlotData is a chart
// with data in it.  What it needs is something to answer the dialogs, which is
// what the driver below is -- it fills the (now named) controls of whichever
// dialog appears, accepts it, and records the text of the report that follows.

#include "chartviewer.h"

#include "chartdialogs.h"
#include "plotdata.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QSpinBox>
#include <QTimer>

#include <cmath>

namespace {

// Answers the dialogs postProcess() raises, in the order it raises them, and
// keeps what they said.
//
// A modal dialog spins its own event loop, so nothing in the test body runs
// until it is dismissed; a timer polling for the active modal is the only way
// in.  Every dialog gets an answer -- an unanswered one stalls the test rather
// than failing it.
class Driver : public QObject {
public:
    PostProcessSpec request;   ///< what to put into the analysis dialog
    int atomsPerCell = 1;      ///< what to put into the Birch-Murnaghan setup
    QStringList reports;       ///< the text of every message box that appeared
    QStringList eosValues;     ///< v0, a0, e0, b0, b0', rms from the EOS result
    int dialogs = 0;           ///< how many modals were answered

    explicit Driver(int budgetMs = 4000) : left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Driver::poll);
        timer.start();
    }

    /// true when a message box carried @p needle
    [[nodiscard]] bool reported(const QString &needle) const
    {
        for (const auto &r : reports)
            if (r.contains(needle)) return true;
        return false;
    }

private:
    void poll()
    {
        if ((left -= 5) < 0) {
            timer.stop();
            return;
        }
        auto *m = QApplication::activeModalWidget();
        if (!m) return;
        ++dialogs;

        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            reports << box->text() + "\n" + box->informativeText();
            box->accept();
            return;
        }
        if (auto *pp = qobject_cast<PostProcessDialog *>(m)) {
            fill(pp);
            pp->accept();
            return;
        }
        if (auto *eos = qobject_cast<EosSetupDialog *>(m)) {
            eos->findChild<QSpinBox *>("atoms")->setValue(atomsPerCell);
            eos->accept();
            return;
        }
        if (auto *res = qobject_cast<EosResultDialog *>(m)) {
            for (const char *n : {"v0", "a0", "e0", "b0", "b0prime", "rms"})
                if (auto *l = res->findChild<QLabel *>(QLatin1String(n))) eosValues << l->text();
            res->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }

    void fill(PostProcessDialog *d) const
    {
        d->findChild<QComboBox *>("analysis")->setCurrentIndex(request.analysis);
        d->findChild<QSpinBox *>("param")->setValue(request.param);
        d->findChild<QLineEdit *>("expression")->setText(request.expression);
        d->findChild<QLineEdit *>("parameters")->setText(request.parameters);
        d->findChild<QLineEdit *>("label")->setText(request.label);
        if (request.fitFrom != request.fitTo) {
            d->findChild<QDoubleSpinBox *>("fitFrom")->setValue(request.fitFrom);
            d->findChild<QDoubleSpinBox *>("fitTo")->setValue(request.fitTo);
        }
    }

    QTimer timer;
    int left;
};

// A chart of y = f(x) over [0, n), which each analysis below reads.
PlotData series(int n, const std::function<double(double)> &f, const QString &yname = "Temp")
{
    PlotData d;
    d.setColumnNames({"Step", yname});
    for (int i = 0; i < n; ++i) {
        const double x = i;
        d.appendRow({x, f(x)});
    }
    return d;
}

class Analysis : public ::testing::Test {
protected:
    void SetUp() override
    {
        win = new ChartWindow(QString(), nullptr);
        win->resize(600, 400);
    }
    void TearDown() override { delete win; }

    void load(const PlotData &d) { win->loadData(d, 0, {1}); }

    /// run the analysis @p driver is set up for, and wait for it to finish
    void run(Driver &driver)
    {
        QMetaObject::invokeMethod(win, "postProcess");
        QCoreApplication::processEvents();
    }

    ChartViewer *view() const { return win->findChild<ChartViewer *>("chartView"); }

    /// the label of the processed-series slot, which each fit renames
    QString processedLabel() const
    {
        auto *smooth = win->findChild<QComboBox *>("smooth");
        return smooth ? smooth->itemText(1) : QString();
    }

    ChartWindow *win = nullptr;
};

} // namespace

// ---------------------------------------------------------------- refusals

TEST_F(Analysis, RefusesAChartWithTooFewPoints)
{
    load(series(1, [](double x) { return x; }));
    Driver d;
    run(d);
    EXPECT_TRUE(d.reported("Not enough data points")) << d.reports.join(" | ").toStdString();
    EXPECT_EQ(d.dialogs, 1) << "the analysis dialog was offered for a chart it cannot analyze";
}

TEST_F(Analysis, DoesNothingWithNoChartAtAll)
{
    Driver d;
    run(d);
    EXPECT_EQ(d.dialogs, 0) << "an analysis was offered with no chart selected";
}

// ---------------------------------------------------------------- autocorrelation

TEST_F(Analysis, AutocorrelationOpensItsOwnWindow)
{
    // the abscissa becomes lag, so the result cannot share the chart's axes and
    // gets a window of its own
    load(series(64, [](double x) { return std::sin(x / 4.0); }));
    const int before = QApplication::topLevelWidgets().size();

    Driver d;
    d.request.analysis = PostProcessSpec::Autocorrelation;
    d.request.param    = 20;
    run(d);

    ChartWindow *acf = nullptr;
    for (auto *w : QApplication::topLevelWidgets())
        if (auto *c = qobject_cast<ChartWindow *>(w); c && c != win) acf = c;
    ASSERT_NE(acf, nullptr) << "no autocorrelation window appeared";
    EXPECT_GT(QApplication::topLevelWidgets().size(), before);
    EXPECT_EQ(acf->numCharts(), 1);
    EXPECT_TRUE(acf->hasTitle("ACF: Temp", 0)) << "the ACF chart is not named after its series";
    EXPECT_EQ(acf->getStep(), 20) << "the lag axis does not run to the requested maximum";
    acf->close();
}

TEST_F(Analysis, AConstantSeriesHasNoAutocorrelation)
{
    load(series(32, [](double) { return 7.0; }));
    Driver d;
    d.request.analysis = PostProcessSpec::Autocorrelation;
    d.request.param    = 8;
    run(d);
    EXPECT_TRUE(d.reported("Could not compute the autocorrelation"))
        << d.reports.join(" | ").toStdString();
}

// ---------------------------------------------------------------- polynomial

TEST_F(Analysis, PolynomialFitOverlaysACurveAndReportsItsCoefficients)
{
    // y = 2x + 3 exactly, so a degree-1 fit must recover it
    load(series(20, [](double x) { return 2.0 * x + 3.0; }));

    Driver d;
    d.request.analysis = PostProcessSpec::Polynomial;
    d.request.param    = 1;
    run(d);

    ASSERT_NE(view(), nullptr);
    EXPECT_TRUE(view()->isEosFit()) << "no fit curve was put on the chart";
    EXPECT_EQ(processedLabel(), "Poly deg 1") << "the processed slot was not renamed";
    EXPECT_TRUE(d.reported("Polynomial fit of degree 1")) << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("c[0] = 3")) << "the constant term was not recovered";
    EXPECT_TRUE(d.reported("c[1] = 2")) << "the slope was not recovered";
}

TEST_F(Analysis, ThePolynomialFitHonoursTheXRange)
{
    // a kink at x=10: fitting only the left half must not see the right
    load(series(20, [](double x) { return x < 10.0 ? x : 100.0 - 9.0 * x; }));

    Driver d;
    d.request.analysis = PostProcessSpec::Polynomial;
    d.request.param    = 1;
    d.request.fitFrom  = 0.0;
    d.request.fitTo    = 9.0;
    run(d);
    EXPECT_TRUE(d.reported("c[1] = 1")) << "the fit used data outside the range: "
                                        << d.reports.join(" | ").toStdString();
}

TEST_F(Analysis, AnEmptyXRangeFallsBackToTheWholeSeriesWithAWarning)
{
    load(series(20, [](double x) { return 2.0 * x; }));
    Driver d;
    d.request.analysis = PostProcessSpec::Polynomial;
    d.request.param    = 1;
    d.request.fitFrom  = 1000.0; // nothing lives here
    d.request.fitTo    = 2000.0;
    run(d);
    EXPECT_TRUE(d.reported("Fewer than 2 data points in the selected x-range"))
        << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(view()->isEosFit()) << "the fallback did not go on to fit anything";
}

// ---------------------------------------------------------------- custom function

TEST_F(Analysis, ACustomFunctionIsPlottedOverTheDataRange)
{
    load(series(20, [](double x) { return x; }));

    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFunction;
    d.request.expression = "2*x + 1";
    run(d);

    EXPECT_TRUE(view()->isEosFit()) << "the function was not drawn on the chart";
    EXPECT_EQ(processedLabel(), "Custom f(x)");
    EXPECT_TRUE(d.reported("Plotted f(x) = 2*x + 1")) << d.reports.join(" | ").toStdString();
}

TEST_F(Analysis, AnUnparseableExpressionIsRefusedWithItsReason)
{
    load(series(20, [](double x) { return x; }));
    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFunction;
    d.request.expression = "2*x +";
    run(d);
    EXPECT_TRUE(d.reported("Could not evaluate the expression"))
        << d.reports.join(" | ").toStdString();
    EXPECT_FALSE(view()->isEosFit()) << "a curve was drawn from an expression that did not parse";
}

// ---------------------------------------------------------------- custom fit

TEST_F(Analysis, ACustomFitRecoversTheParametersItWasGivenAGuessFor)
{
    // y = 3*exp(-0.2x), fitted from a deliberately wrong starting point
    load(series(40, [](double x) { return 3.0 * std::exp(-0.2 * x); }));

    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFit;
    d.request.expression = "a*exp(-b*x)";
    d.request.parameters = "a=1, b=0.5";
    d.request.label      = "decay";
    run(d);

    EXPECT_TRUE(view()->isEosFit()) << "the fitted curve was not drawn";
    EXPECT_EQ(processedLabel(), "decay") << "the fit did not take the label it was given";
    EXPECT_TRUE(d.reported("Custom fit of f(x) = a*exp(-b*x)"))
        << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("a = 3")) << "the amplitude was not recovered: "
                                     << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("b = 0.2")) << "the rate was not recovered";
}

TEST_F(Analysis, ACustomFitWithoutParametersSaysWhatItWanted)
{
    load(series(20, [](double x) { return x; }));
    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFit;
    d.request.expression = "a*x";
    d.request.parameters = "not a parameter list";
    run(d);
    EXPECT_TRUE(d.reported("name=guess pairs")) << d.reports.join(" | ").toStdString();
}

TEST_F(Analysis, AFitThatCannotConvergeSaysSo)
{
    load(series(20, [](double x) { return x; }));
    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFit;
    d.request.expression = "a*nosuchfunction(x)";
    d.request.parameters = "a=1";
    run(d);
    EXPECT_TRUE(d.reported("The fit could not be completed"))
        << d.reports.join(" | ").toStdString();
}

TEST_F(Analysis, AnUnlabelledCustomFitIsNamedAfterItsExpression)
{
    load(series(30, [](double x) { return 2.0 * x + 1.0; }));
    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFit;
    d.request.expression = "a*x+b";
    d.request.parameters = "a=1, b=0";
    run(d);
    EXPECT_EQ(processedLabel(), "a*x+b") << "a short expression should name the slot itself";
}

TEST_F(Analysis, ALongFitNameIsShortenedForTheSlot)
{
    load(series(30, [](double x) { return 2.0 * x + 1.0; }));
    Driver d;
    d.request.analysis   = PostProcessSpec::CustomFit;
    d.request.expression = "a*x+b";
    d.request.parameters = "a=1, b=0";
    d.request.label      = "a considerably longer name than fits";
    run(d);
    EXPECT_EQ(processedLabel(), "Custom fit") << "a long label was pasted into the combo whole";
}

// ---------------------------------------------------------------- block average

TEST_F(Analysis, BlockAveragingReportsAMeanAndMarksItOnTheChart)
{
    // a noisy series about a known mean
    load(series(400, [](double x) { return 10.0 + std::sin(x) + 0.25 * std::cos(x / 3.0); }));

    Driver d;
    d.request.analysis = PostProcessSpec::BlockAverage;
    d.request.param    = 20;
    run(d);

    EXPECT_TRUE(d.reported("mean")) << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("std. error")) << "the block-averaged error was not reported";
    EXPECT_TRUE(d.reported("tau_int")) << "the integrated autocorrelation time was not reported";
    EXPECT_TRUE(d.reported("N_eff")) << "the effective sample count was not reported";
}

TEST_F(Analysis, BlockAveragingAConstantSeriesSaysItCannot)
{
    load(series(100, [](double) { return 4.0; }));
    Driver d;
    d.request.analysis = PostProcessSpec::BlockAverage;
    d.request.param    = 10;
    run(d);
    EXPECT_TRUE(d.reported("Could not analyze the series"))
        << d.reports.join(" | ").toStdString();
}

// ---------------------------------------------------------------- steady state

TEST_F(Analysis, SteadyStateDetectionFindsABurnInAndReportsIt)
{
    // a transient that settles: high for the first quarter, flat after
    load(series(400, [](double x) {
        return x < 100.0 ? 50.0 - 0.4 * x : 10.0 + 0.5 * std::sin(x);
    }));

    Driver d;
    d.request.analysis = PostProcessSpec::SteadyState;
    run(d);

    EXPECT_TRUE(d.reported("burn-in cutoff")) << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("steady mean")) << "the post-burn-in mean was not reported";
    EXPECT_TRUE(d.reported("samples kept")) << "how much data survived was not reported";
}

TEST_F(Analysis, SteadyStateDetectionOnTooShortASeriesSaysSo)
{
    load(series(3, [](double x) { return x; }));
    Driver d;
    d.request.analysis = PostProcessSpec::SteadyState;
    run(d);
    EXPECT_TRUE(d.reported("Could not analyze the series"))
        << d.reports.join(" | ").toStdString();
}

// ---------------------------------------------------------------- Birch-Murnaghan

TEST_F(Analysis, TheEosFitAsksForTheAtomCountThenReportsTheLatticeConstant)
{
    // a cohesive-energy curve with a minimum inside the data, which is what the
    // fit needs; volumes must be positive
    PlotData d;
    d.setColumnNames({"Volume", "PotEng"});
    for (int i = 0; i < 24; ++i) {
        const double v = 12.0 + 0.5 * i;
        const double e = -4.0 + 0.004 * (v - 16.0) * (v - 16.0);
        d.appendRow({v, e});
    }
    win->loadData(d, 0, {1});

    Driver drv;
    drv.request.analysis = PostProcessSpec::Eos;
    drv.atomsPerCell     = 4; // FCC
    run(drv);

    EXPECT_GE(drv.dialogs, 3) << "the setup dialog or the result dialog never appeared";
    EXPECT_EQ(drv.eosValues.size(), 6) << "the result dialog did not report every quantity";
    EXPECT_TRUE(view()->isEosFit()) << "no EOS curve was drawn";
    EXPECT_EQ(processedLabel(), "EOS fit");

    // V0 should land near the minimum of the parabola, and a0 = cbrt(4 * V0)
    ASSERT_FALSE(drv.eosValues.isEmpty());
    const double v0 = drv.eosValues.at(0).toDouble();
    const double a0 = drv.eosValues.at(1).toDouble();
    EXPECT_NEAR(v0, 16.0, 1.0) << "the equilibrium volume is not near the curve's minimum";
    EXPECT_NEAR(a0, std::cbrt(4.0 * v0), 1.0e-3) << "a0 does not follow from V0 and N";
}

TEST_F(Analysis, TheEosFitRefusesDataWithNoMinimumInIt)
{
    // a monotonic curve: no equilibrium volume to find
    PlotData d;
    d.setColumnNames({"Volume", "PotEng"});
    for (int i = 0; i < 20; ++i)
        d.appendRow({10.0 + i, -1.0 * i});
    win->loadData(d, 0, {1});

    Driver drv;
    drv.request.analysis = PostProcessSpec::Eos;
    run(drv);
    EXPECT_TRUE(drv.reported("Birch-Murnaghan fit failed"))
        << drv.reports.join(" | ").toStdString();
    EXPECT_FALSE(view()->isEosFit()) << "a curve was drawn from a fit that failed";
}

TEST_F(Analysis, CancellingTheAnalysisLeavesTheChartAlone)
{
    load(series(20, [](double x) { return 2.0 * x; }));

    // a driver that rejects everything it is shown
    class Rejecter : public QObject {
    public:
        Rejecter()
        {
            timer.setInterval(5);
            connect(&timer, &QTimer::timeout, this, [this]() {
                if ((left -= 5) < 0) { timer.stop(); return; }
                if (auto *m = QApplication::activeModalWidget()) {
                    if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
                    ++seen;
                }
            });
            timer.start();
        }
        int seen = 0;

    private:
        QTimer timer;
        int left = 2000;
    } rejecter;

    QMetaObject::invokeMethod(win, "postProcess");
    QCoreApplication::processEvents();

    EXPECT_GE(rejecter.seen, 1) << "no dialog was raised to cancel";
    EXPECT_FALSE(view()->isEosFit()) << "cancelling still put a fit on the chart";
    EXPECT_EQ(win->numCharts(), 1);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_chartanalysis.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
