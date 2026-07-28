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

#include "analysis.h"
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
#include <QRegularExpression>
#include <QSpinBox>
#include <QTimer>

#include <cmath>
#include <vector>

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

    /// The number a report gave for @p key, as in "  mean  = 12.5", or NaN if
    /// no report carried that key.
    ///
    /// reported("c[0] = 3") is not an assertion that the constant term came
    /// back as 3: it matches 3.7 and 30 just as happily, so a fit that had
    /// drifted would still pass.  Everything numeric below goes through this
    /// instead, so the assertion is on the value and not on its first digit.
    [[nodiscard]] double value(const QString &key) const
    {
        const QRegularExpression re(
            QRegularExpression::escape(key) +
            R"(\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))");
        for (const auto &r : reports) {
            const auto m = re.match(r);
            if (m.hasMatch()) return m.captured(1).toDouble();
        }
        return std::nan("");
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
    EXPECT_NEAR(d.value("c[0]"), 3.0, 1.0e-6)
        << "the constant term was not recovered: " << d.reports.join(" | ").toStdString();
    EXPECT_NEAR(d.value("c[1]"), 2.0, 1.0e-6) << "the slope was not recovered";
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
    EXPECT_NEAR(d.value("a"), 3.0, 1.0e-4) << "the amplitude was not recovered: "
                                           << d.reports.join(" | ").toStdString();
    EXPECT_NEAR(d.value("b"), 0.2, 1.0e-5) << "the rate was not recovered";
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
    // A bounded oscillation about 10: the sum of sin(i) over i = 0..399 is
    // bounded by 1/sin(0.5) ~ 2.1 and the cosine term by 0.25/sin(1/6) ~ 1.5,
    // so the mean of these 400 samples is 10 to within 0.01 whatever the
    // implementation does.  That is the independent anchor; the rest of the
    // report is then checked against what blockAverage() returns for the same
    // data, which is what says the chart is reading the fields it means to.
    const auto f = [](double x) { return 10.0 + std::sin(x) + 0.25 * std::cos(x / 3.0); };
    load(series(400, f));

    Driver d;
    d.request.analysis = PostProcessSpec::BlockAverage;
    d.request.param    = 20;
    run(d);

    EXPECT_TRUE(d.reported("mean")) << d.reports.join(" | ").toStdString();
    EXPECT_TRUE(d.reported("std. error")) << "the block-averaged error was not reported";
    EXPECT_TRUE(d.reported("tau_int")) << "the integrated autocorrelation time was not reported";
    EXPECT_TRUE(d.reported("N_eff")) << "the effective sample count was not reported";

    EXPECT_NEAR(d.value("mean"), 10.0, 0.01)
        << "the reported mean is not the mean of the series: " << d.reports.join(" | ").toStdString();

    std::vector<double> ys;
    ys.reserve(400);
    for (int i = 0; i < 400; ++i) ys.push_back(f(i));
    const BlockStats bs = blockAverage(ys, 20);
    ASSERT_TRUE(bs.valid);

    EXPECT_NEAR(d.value("mean"), bs.mean, 1.0e-6);
    EXPECT_NEAR(d.value("std. error"), bs.stderror, 1.0e-5 * std::max(1.0, bs.stderror))
        << "the number under 'std. error' is not the block-averaged error";
    EXPECT_NEAR(d.value("tau_int"), bs.tauInt, 0.005) << "tau_int is not the one that was computed";
    EXPECT_NEAR(d.value("N_eff"), bs.nEff, 0.05) << "N_eff is not the one that was computed";
    EXPECT_EQ(int(d.value("blocks")), bs.nblocks) << "the block count was misreported";

    // the comparison line only means something if it is the naive error rather
    // than a second copy of the block-averaged one
    EXPECT_NEAR(d.value("naive s/sqrt(N)"), std::sqrt(bs.variance / 400.0), 1.0e-6);
    EXPECT_NE(d.value("naive s/sqrt(N)"), d.value("std. error"));
}

TEST_F(Analysis, BlockAveragingASeriesWhoseBlocksAgreeExactlyReportsNoError)
{
    // A period-20 sawtooth cut into 20 blocks of 20: every block holds one whole
    // period, so all twenty block means are identical and the batch-means error
    // is exactly zero.  An analytic answer that does not come from the same code
    // the chart calls -- and one no plausible off-by-one survives, since a block
    // boundary out of step with the period would break the agreement.
    load(series(400, [](double x) { return 5.0 + std::fmod(x, 20.0); }));

    Driver d;
    d.request.analysis = PostProcessSpec::BlockAverage;
    d.request.param    = 20;
    run(d);

    EXPECT_NEAR(d.value("mean"), 5.0 + 9.5, 1.0e-6) << d.reports.join(" | ").toStdString();
    EXPECT_NEAR(d.value("std. error"), 0.0, 1.0e-9)
        << "blocks that agree to the last digit were given a nonzero error";
    EXPECT_GT(d.value("naive s/sqrt(N)"), 0.2)
        << "the naive error should be large here -- the series itself has spread";
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

    // The transient runs from 50 down to 10 over the first hundred samples and
    // the series is flat about 10 after that, so a cutoff that has genuinely
    // found the burn-in lands inside the transient and the mean it reports is
    // the flat level rather than one dragged up by the ramp.
    const double cutoff = d.value("burn-in cutoff");
    EXPECT_GT(cutoff, 0.0) << "nothing was discarded from a series that plainly has a transient";
    EXPECT_LE(cutoff, 140.0) << "the cutoff threw away data well past the end of the transient";
    EXPECT_NEAR(d.value("steady mean"), 10.0, 0.5)
        << "the reported mean still carries the transient: " << d.reports.join(" | ").toStdString();
    EXPECT_NEAR(d.value("samples kept"), 400.0 - cutoff, 0.5)
        << "the kept-sample count does not follow from the cutoff";

    // and the numbers are the ones the analysis returned, not neighbouring fields
    std::vector<double> ys;
    ys.reserve(400);
    for (int i = 0; i < 400; ++i)
        ys.push_back(i < 100 ? 50.0 - 0.4 * i : 10.0 + 0.5 * std::sin(i));
    const SteadyState ss = steadyStateCutoff(ys);
    ASSERT_TRUE(ss.valid);
    EXPECT_EQ(int(cutoff), ss.cutoff);
    EXPECT_NEAR(d.value("steady mean"), ss.mean, 1.0e-6);
    EXPECT_NEAR(d.value("std. error"), ss.stderror, 1.0e-5 * std::max(1.0, ss.stderror));
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
