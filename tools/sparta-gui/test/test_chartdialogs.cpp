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

// The chart window's four modal dialogs.
//
// Each was built on the stack inside a ChartWindow method, reading a live
// ChartViewer as it went and writing back after exec() returned -- so none
// could be constructed without a chart with data in it, and the only thing ever
// checked about them was that opening one did not crash.  Split out, each is a
// pure function of the plain struct it is handed, and the mapping between its
// controls and that struct is what these check.

#include "chartdialogs.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QSpinBox>

#include <cmath>

namespace {

template <class W> W *ctl(const QDialog &d, const char *name)
{
    auto *w = d.findChild<W *>(QLatin1String(name));
    EXPECT_NE(w, nullptr) << "no control named " << name;
    return w;
}

// a style with every field set to something distinguishable from the defaults
ChartStyle populatedStyle()
{
    ChartStyle s;
    s.rawMode       = ChartDisplayMode::Points;
    s.rawColor      = QColor(10, 20, 30);
    s.rawWidth      = 4.5;
    s.rawPointSize  = 12.0;
    s.procMode      = ChartDisplayMode::LinesAndPoints;
    s.procColor     = QColor(200, 100, 50);
    s.procWidth     = 2.5;
    s.procPointSize = 6.0;
    s.legend        = LegendPos::BottomLeft;
    return s;
}

RefLine line(RefOrient o, double v, const QString &label, const QColor &c, RefAnchor a)
{
    RefLine r;
    r.orient = o;
    r.value  = v;
    r.label  = label;
    r.color  = c;
    r.anchor = a;
    return r;
}

} // namespace

/* ==================================================================== */
/*  Chart style                                                          */
/* ==================================================================== */

TEST(ChartStyleDlg, RoundTripsAFullyPopulatedStyle)
{
    // build from a style, change nothing, read it back: a control wired to the
    // wrong field cannot survive this
    const ChartStyle in = populatedStyle();
    ChartStyleDialog d(in);
    const ChartStyle out = d.style();

    EXPECT_EQ(out.rawMode, in.rawMode);
    EXPECT_EQ(out.rawColor, in.rawColor);
    EXPECT_DOUBLE_EQ(out.rawWidth, in.rawWidth);
    EXPECT_DOUBLE_EQ(out.rawPointSize, in.rawPointSize);
    EXPECT_EQ(out.procMode, in.procMode);
    EXPECT_EQ(out.procColor, in.procColor);
    EXPECT_DOUBLE_EQ(out.procWidth, in.procWidth);
    EXPECT_DOUBLE_EQ(out.procPointSize, in.procPointSize);
    EXPECT_EQ(out.legend, in.legend);
}

TEST(ChartStyleDlg, ReadingItBackIsRepeatable)
{
    ChartStyleDialog d(populatedStyle());
    const ChartStyle a = d.style();
    const ChartStyle b = d.style();
    EXPECT_EQ(a.rawMode, b.rawMode);
    EXPECT_DOUBLE_EQ(a.rawWidth, b.rawWidth);
    EXPECT_EQ(a.legend, b.legend);
}

TEST(ChartStyleDlg, TheRawAndProcessedSectionsAreSeparateControls)
{
    // the two sections are built from the same three helpers, so a copy-paste
    // slip that points both at the raw widgets would go unnoticed
    ChartStyleDialog d(ChartStyle{});
    ctl<QComboBox>(d, "rawMode")->setCurrentIndex(1);       // Points
    ctl<QComboBox>(d, "procMode")->setCurrentIndex(2);      // Lines + Points
    ctl<QDoubleSpinBox>(d, "rawWidth")->setValue(1.5);
    ctl<QDoubleSpinBox>(d, "procWidth")->setValue(7.5);
    ctl<QDoubleSpinBox>(d, "rawPointSize")->setValue(3.0);
    ctl<QDoubleSpinBox>(d, "procPointSize")->setValue(20.0);

    const ChartStyle s = d.style();
    EXPECT_EQ(s.rawMode, ChartDisplayMode::Points);
    EXPECT_EQ(s.procMode, ChartDisplayMode::LinesAndPoints);
    EXPECT_DOUBLE_EQ(s.rawWidth, 1.5);
    EXPECT_DOUBLE_EQ(s.procWidth, 7.5);
    EXPECT_DOUBLE_EQ(s.rawPointSize, 3.0);
    EXPECT_DOUBLE_EQ(s.procPointSize, 20.0);
}

TEST(ChartStyleDlg, EveryDisplayModeSurvivesTheTrip)
{
    for (auto mode : {ChartDisplayMode::Lines, ChartDisplayMode::Points,
                      ChartDisplayMode::LinesAndPoints}) {
        ChartStyle in;
        in.rawMode  = mode;
        in.procMode = mode;
        ChartStyleDialog d(in);
        EXPECT_EQ(d.style().rawMode, mode);
        EXPECT_EQ(d.style().procMode, mode);
    }
}

TEST(ChartStyleDlg, EveryLegendPlacementSurvivesTheTrip)
{
    for (auto pos : {LegendPos::Off, LegendPos::TopLeft, LegendPos::TopRight,
                     LegendPos::BottomLeft, LegendPos::BottomRight}) {
        ChartStyle in;
        in.legend = pos;
        ChartStyleDialog d(in);
        EXPECT_EQ(d.style().legend, pos) << "legend placement " << static_cast<int>(pos);
    }
}

TEST(ChartStyleDlg, AnUnsetColourBecomesTheDefaultRatherThanStayingInvalid)
{
    // "no colour" means the theme default; the dialog has to show something,
    // and whatever it shows is what the chart will be given on accept
    ChartStyleDialog d(ChartStyle{}); // both colours default-constructed, i.e. invalid
    const ChartStyle s = d.style();
    EXPECT_TRUE(s.rawColor.isValid()) << "an invalid colour was handed straight back";
    EXPECT_TRUE(s.procColor.isValid());
    EXPECT_NE(s.rawColor, s.procColor) << "the raw and processed defaults are the same colour";
}

TEST(ChartStyleDlg, TheColourButtonsShowTheColourTheyHold)
{
    ChartStyleDialog d(populatedStyle());
    EXPECT_EQ(ctl<QPushButton>(d, "rawColor")->text(), QColor(10, 20, 30).name());
    EXPECT_EQ(ctl<QPushButton>(d, "procColor")->text(), QColor(200, 100, 50).name());
}

TEST(ChartStyleDlg, TheWidthAndPointSizeRangesAreEnforced)
{
    ChartStyle in;
    in.rawWidth     = 1000.0; // far above the maximum
    in.rawPointSize = 0.0;    // below the minimum
    ChartStyleDialog d(in);
    const ChartStyle s = d.style();
    EXPECT_LE(s.rawWidth, 20.0) << "a line width past the maximum was accepted";
    EXPECT_GE(s.rawPointSize, 1.0) << "a marker smaller than a pixel was accepted";
}

/* ==================================================================== */
/*  Postprocess                                                          */
/* ==================================================================== */

TEST(PostProcessDlg, OffersEveryAnalysis)
{
    PostProcessDialog d(100, 0.0, 99.0);
    auto *box = ctl<QComboBox>(d, "analysis");
    EXPECT_EQ(box->count(), 7) << "an analysis was added or lost";
    EXPECT_EQ(box->itemText(PostProcessSpec::Autocorrelation), "Autocorrelation");
    EXPECT_EQ(box->itemText(PostProcessSpec::Polynomial), "Polynomial fit");
    EXPECT_EQ(box->itemText(PostProcessSpec::Eos), "Birch-Murnaghan EOS fit");
    EXPECT_EQ(box->itemText(PostProcessSpec::CustomFunction), "Custom function");
    EXPECT_EQ(box->itemText(PostProcessSpec::CustomFit), "Custom fit");
    EXPECT_EQ(box->itemText(PostProcessSpec::BlockAverage), "Block-average uncertainty");
    EXPECT_EQ(box->itemText(PostProcessSpec::SteadyState), "Steady-state detection");
}

TEST(PostProcessDlg, OpensOnAutocorrelationWithAHalfSeriesLag)
{
    PostProcessDialog d(100, 0.0, 99.0);
    const PostProcessSpec s = d.spec();
    EXPECT_EQ(s.analysis, PostProcessSpec::Autocorrelation);
    EXPECT_EQ(s.param, 50) << "the default max lag is not half the series";
}

TEST(PostProcessDlg, TheFitRangeDefaultsToTheDataRange)
{
    PostProcessDialog d(100, -12.5, 87.25);
    ctl<QComboBox>(d, "analysis")->setCurrentIndex(PostProcessSpec::Polynomial);
    const PostProcessSpec s = d.spec();
    EXPECT_DOUBLE_EQ(s.fitFrom, -12.5);
    EXPECT_DOUBLE_EQ(s.fitTo, 87.25);
}

TEST(PostProcessDlg, ThePolynomialDegreeCannotExceedWhatTheDataSupports)
{
    // a degree-n polynomial needs n+1 points; with four points the highest
    // fittable degree is three
    PostProcessDialog d(4, 0.0, 3.0);
    ctl<QComboBox>(d, "analysis")->setCurrentIndex(PostProcessSpec::Polynomial);
    auto *spin = ctl<QSpinBox>(d, "param");
    EXPECT_EQ(spin->maximum(), 3) << "a degree with more coefficients than points was offered";

    // and it is capped at 8 however much data there is
    PostProcessDialog big(10000, 0.0, 1.0);
    ctl<QComboBox>(big, "analysis")->setCurrentIndex(PostProcessSpec::Polynomial);
    EXPECT_EQ(ctl<QSpinBox>(big, "param")->maximum(), 8);
}

TEST(PostProcessDlg, TheMaxLagCannotReachPastTheSeries)
{
    PostProcessDialog d(10, 0.0, 9.0);
    EXPECT_EQ(ctl<QSpinBox>(d, "param")->maximum(), 9);
}

TEST(PostProcessDlg, TheBlockCountStartsAtTheSquareRootOfTheSeries)
{
    // the usual batch-means rule of thumb: sqrt(N) blocks
    PostProcessDialog d(100, 0.0, 99.0);
    ctl<QComboBox>(d, "analysis")->setCurrentIndex(PostProcessSpec::BlockAverage);
    auto *spin = ctl<QSpinBox>(d, "param");
    EXPECT_EQ(spin->value(), 10);
    EXPECT_EQ(spin->minimum(), 2) << "one block is not an average of anything";
    EXPECT_EQ(spin->maximum(), 50) << "a block needs at least two samples in it";
}

TEST(PostProcessDlg, TheParameterIsRelabelledPerAnalysis)
{
    PostProcessDialog d(100, 0.0, 99.0);
    auto *box = ctl<QComboBox>(d, "analysis");
    auto label = [&d]() {
        for (auto *l : d.findChildren<QLabel *>())
            if (l->text().endsWith(':') && l->isVisibleTo(&d)) {
                const QString t = l->text();
                if (t == "Max lag:" || t == "Degree:" || t == "Blocks:") return t;
            }
        return QString();
    };
    EXPECT_EQ(label(), "Max lag:");
    box->setCurrentIndex(PostProcessSpec::Polynomial);
    EXPECT_EQ(label(), "Degree:");
    box->setCurrentIndex(PostProcessSpec::BlockAverage);
    EXPECT_EQ(label(), "Blocks:");
}

TEST(PostProcessDlg, OnlyTheAnalysesThatNeedAnExpressionShowOne)
{
    PostProcessDialog d(100, 0.0, 99.0);
    d.show(); // isVisible() on a child needs the dialog itself to be shown
    auto *box  = ctl<QComboBox>(d, "analysis");
    auto *expr = ctl<QLineEdit>(d, "expression");

    for (int i = 0; i < box->count(); ++i) {
        box->setCurrentIndex(i);
        const bool wanted = (i == PostProcessSpec::CustomFunction) ||
                            (i == PostProcessSpec::CustomFit);
        EXPECT_EQ(expr->isVisible(), wanted) << "analysis " << box->itemText(i).toStdString();
    }
}

TEST(PostProcessDlg, OnlyTheCustomFitAsksForParametersAndALabel)
{
    PostProcessDialog d(100, 0.0, 99.0);
    d.show();
    auto *box    = ctl<QComboBox>(d, "analysis");
    auto *params = ctl<QLineEdit>(d, "parameters");
    auto *label  = ctl<QLineEdit>(d, "label");

    for (int i = 0; i < box->count(); ++i) {
        box->setCurrentIndex(i);
        const bool wanted = (i == PostProcessSpec::CustomFit);
        EXPECT_EQ(params->isVisible(), wanted) << "analysis " << i;
        EXPECT_EQ(label->isVisible(), wanted) << "analysis " << i;
    }
}

TEST(PostProcessDlg, OnlyTheFittingAnalysesShowAFitRange)
{
    PostProcessDialog d(100, 0.0, 99.0);
    d.show();
    auto *box  = ctl<QComboBox>(d, "analysis");
    auto *from = ctl<QDoubleSpinBox>(d, "fitFrom");

    for (int i = 0; i < box->count(); ++i) {
        box->setCurrentIndex(i);
        const bool wanted = (i >= PostProcessSpec::Polynomial) && (i <= PostProcessSpec::CustomFit);
        EXPECT_EQ(from->isVisible(), wanted) << "analysis " << box->itemText(i).toStdString();
        // and the struct agrees with the dialog about which those are
        EXPECT_EQ(d.spec().usesFitRange(), wanted) << "analysis " << i;
    }
}

TEST(PostProcessDlg, TheTextFieldsReachTheSpec)
{
    PostProcessDialog d(100, 0.0, 99.0);
    ctl<QComboBox>(d, "analysis")->setCurrentIndex(PostProcessSpec::CustomFit);
    ctl<QLineEdit>(d, "expression")->setText("  a*exp(-b*x)  ");
    ctl<QLineEdit>(d, "parameters")->setText("a=1, b=0.5");
    ctl<QLineEdit>(d, "label")->setText("  decay  ");
    ctl<QDoubleSpinBox>(d, "fitFrom")->setValue(10.0);
    ctl<QDoubleSpinBox>(d, "fitTo")->setValue(80.0);

    const PostProcessSpec s = d.spec();
    EXPECT_EQ(s.analysis, PostProcessSpec::CustomFit);
    EXPECT_EQ(s.expression, "a*exp(-b*x)") << "the expression was not trimmed";
    EXPECT_EQ(s.parameters, "a=1, b=0.5") << "the parameters must reach the parser verbatim";
    EXPECT_EQ(s.label, "decay") << "the label was not trimmed";
    EXPECT_DOUBLE_EQ(s.fitFrom, 10.0);
    EXPECT_DOUBLE_EQ(s.fitTo, 80.0);
}

TEST(PostProcessDlg, SurvivesTheSmallestChartItIsOfferedFor)
{
    // ChartWindow refuses fewer than two points, so two is the floor here
    PostProcessDialog d(2, 0.0, 1.0);
    auto *box = ctl<QComboBox>(d, "analysis");
    for (int i = 0; i < box->count(); ++i) {
        box->setCurrentIndex(i);
        const PostProcessSpec s = d.spec();
        EXPECT_GE(s.param, 1) << "analysis " << i << " offered a parameter below its own minimum";
    }
}

TEST(PostProcessDlg, AConstantXRangeIsSurvivable)
{
    PostProcessDialog d(10, 5.0, 5.0);
    ctl<QComboBox>(d, "analysis")->setCurrentIndex(PostProcessSpec::Polynomial);
    const PostProcessSpec s = d.spec();
    EXPECT_DOUBLE_EQ(s.fitFrom, 5.0);
    EXPECT_DOUBLE_EQ(s.fitTo, 5.0);
}

/* ==================================================================== */
/*  Birch-Murnaghan setup and result                                     */
/* ==================================================================== */

TEST(EosDlg, ShowsWhichColumnsItWillTreatAsVolumeAndEnergy)
{
    EosSetupDialog d("Volume", "PotEng");
    EXPECT_TRUE(ctl<QLabel>(d, "xLabel")->text().contains("Volume"));
    EXPECT_TRUE(ctl<QLabel>(d, "yLabel")->text().contains("PotEng"));
}

TEST(EosDlg, DefaultsToOneAtomPerCell)
{
    // N=1 means the x axis is already the conventional cell volume, which is
    // the assumption that does not silently rescale the lattice constant
    EosSetupDialog d("V", "E");
    EXPECT_EQ(d.atomsPerCell(), 1);
}

TEST(EosDlg, TheAtomCountIsBoundedAndReadBack)
{
    EosSetupDialog d("V", "E");
    auto *spin = ctl<QSpinBox>(d, "atoms");
    EXPECT_EQ(spin->minimum(), 1) << "zero atoms per cell would make the lattice constant zero";
    spin->setValue(4); // FCC
    EXPECT_EQ(d.atomsPerCell(), 4);
    spin->setValue(0);
    EXPECT_EQ(d.atomsPerCell(), 1) << "the lower bound was not enforced";
}

TEST(EosDlg, TheLatticeConstantIsTheCubeRootOfNTimesV0)
{
    EXPECT_DOUBLE_EQ(EosResultDialog::latticeConstant(64.0, 1), 4.0);
    EXPECT_DOUBLE_EQ(EosResultDialog::latticeConstant(16.0, 4), 4.0);
    EXPECT_DOUBLE_EQ(EosResultDialog::latticeConstant(0.0, 4), 0.0);
}

TEST(EosDlg, TheResultReportsEveryFittedQuantity)
{
    EosFit f;
    f.ok      = true;
    f.v0      = 16.0;
    f.e0      = -3.54;
    f.b0      = 0.6789;
    f.b0prime = 4.25;
    f.rms     = 1.5e-4;

    EosResultDialog d(f, 4);
    EXPECT_EQ(ctl<QLabel>(d, "v0")->text(), "16");
    EXPECT_EQ(ctl<QLabel>(d, "a0")->text(), "4") << "the lattice constant of V0=16 with N=4";
    EXPECT_EQ(ctl<QLabel>(d, "e0")->text(), "-3.54");
    EXPECT_TRUE(ctl<QLabel>(d, "b0")->text().startsWith("0.6789"));
    EXPECT_EQ(ctl<QLabel>(d, "b0prime")->text(), "4.25");
    EXPECT_FALSE(ctl<QLabel>(d, "rms")->text().isEmpty());
}

TEST(EosDlg, TheResultNumbersAreSelectable)
{
    // they are meant to be copied into a paper, not read back off the screen
    EosFit f;
    f.v0 = 1.0;
    EosResultDialog d(f, 1);
    EXPECT_TRUE(ctl<QLabel>(d, "v0")->textInteractionFlags() & Qt::TextSelectableByMouse);
}

/* ==================================================================== */
/*  Reference lines                                                      */
/* ==================================================================== */

TEST(RefLinesDlg, RoundTripsTheLinesItWasGiven)
{
    QList<RefLine> in;
    in << line(RefOrient::Vertical, 12.5, "start", QColor(10, 20, 30), RefAnchor::Start)
       << line(RefOrient::Horizontal, -4.25, "mean", QColor(200, 30, 40), RefAnchor::End);

    RefLinesDialog d(in, RefLineStyle{});
    const QList<RefLine> out = d.lines();
    ASSERT_EQ(out.size(), in.size());
    for (int i = 0; i < in.size(); ++i) {
        EXPECT_EQ(out.at(i).orient, in.at(i).orient) << "row " << i;
        EXPECT_DOUBLE_EQ(out.at(i).value, in.at(i).value) << "row " << i;
        EXPECT_EQ(out.at(i).label, in.at(i).label) << "row " << i;
        EXPECT_EQ(out.at(i).color, in.at(i).color) << "row " << i;
        EXPECT_EQ(out.at(i).anchor, in.at(i).anchor) << "row " << i;
    }
}

TEST(RefLinesDlg, RoundTripsTheLabelStyle)
{
    RefLineStyle in;
    in.fontSize = 14.5;
    in.gap      = 12.0;
    in.boxed    = true;

    RefLinesDialog d({}, in);
    const RefLineStyle out = d.labelStyle();
    EXPECT_DOUBLE_EQ(out.fontSize, in.fontSize);
    EXPECT_DOUBLE_EQ(out.gap, in.gap);
    EXPECT_EQ(out.boxed, in.boxed);
}

TEST(RefLinesDlg, StartsEmptyWhenThereAreNoLines)
{
    RefLinesDialog d({}, RefLineStyle{});
    EXPECT_EQ(d.lineCount(), 0);
    EXPECT_TRUE(d.lines().isEmpty());
}

TEST(RefLinesDlg, AddedLinesAppearInTheAnswer)
{
    RefLinesDialog d({}, RefLineStyle{});
    EXPECT_EQ(d.addLine(), 1);
    EXPECT_EQ(d.addLine(), 2);

    ctl<QDoubleSpinBox>(d, "value1")->setValue(7.5);
    ctl<QLineEdit>(d, "label1")->setText("second");

    const QList<RefLine> out = d.lines();
    ASSERT_EQ(out.size(), 2);
    EXPECT_DOUBLE_EQ(out.at(1).value, 7.5);
    EXPECT_EQ(out.at(1).label, "second");
    EXPECT_EQ(out.at(0).orient, RefOrient::Vertical) << "a new line starts vertical";
}

TEST(RefLinesDlg, ARemovedLineIsGoneFromTheAnswer)
{
    // the failure that matters: a row the user deleted that comes back on the
    // chart anyway, because the answer was built from a list the delete missed
    QList<RefLine> in;
    in << line(RefOrient::Vertical, 1.0, "one", QColor(1, 1, 1), RefAnchor::Start)
       << line(RefOrient::Vertical, 2.0, "two", QColor(2, 2, 2), RefAnchor::Start)
       << line(RefOrient::Vertical, 3.0, "three", QColor(3, 3, 3), RefAnchor::Start);

    RefLinesDialog d(in, RefLineStyle{});
    ASSERT_TRUE(d.removeLine(1));
    EXPECT_EQ(d.lineCount(), 2);

    const QList<RefLine> out = d.lines();
    ASSERT_EQ(out.size(), 2);
    EXPECT_EQ(out.at(0).label, "one");
    EXPECT_EQ(out.at(1).label, "three") << "the removed row is still in the answer";
}

TEST(RefLinesDlg, RemovingEveryLineLeavesNone)
{
    QList<RefLine> in;
    in << line(RefOrient::Vertical, 1.0, "one", QColor(1, 1, 1), RefAnchor::Start)
       << line(RefOrient::Horizontal, 2.0, "two", QColor(2, 2, 2), RefAnchor::End);

    RefLinesDialog d(in, RefLineStyle{});
    while (d.lineCount() > 0)
        ASSERT_TRUE(d.removeLine(0));
    EXPECT_TRUE(d.lines().isEmpty());
}

TEST(RefLinesDlg, RemovingAnIndexThatIsNotThereChangesNothing)
{
    RefLinesDialog d({}, RefLineStyle{});
    d.addLine();
    EXPECT_FALSE(d.removeLine(-1));
    EXPECT_FALSE(d.removeLine(1));
    EXPECT_FALSE(d.removeLine(99));
    EXPECT_EQ(d.lineCount(), 1);
}

TEST(RefLinesDlg, RemovingThenAddingKeepsTheRowsAddressable)
{
    // the object names carry a row index; after a removal the surviving rows
    // keep the names they were built with, so a test (or the walker) must still
    // find a control for every row that exists
    RefLinesDialog d({}, RefLineStyle{});
    d.addLine();
    d.addLine();
    ASSERT_TRUE(d.removeLine(0));
    d.addLine();
    EXPECT_EQ(d.lineCount(), 2);
    EXPECT_EQ(d.lines().size(), 2) << "a row exists that lines() cannot read";
}

TEST(RefLinesDlg, TheAnchorItemsFollowTheOrientation)
{
    // a horizontal line's label sits left/right along it, a vertical one's
    // top/bottom -- the same three anchors, named for what the user sees
    RefLinesDialog d({}, RefLineStyle{});
    d.addLine();
    auto *orient = ctl<QComboBox>(d, "orient0");
    auto *anchor = ctl<QComboBox>(d, "anchor0");

    EXPECT_EQ(anchor->itemText(0), "Top");
    EXPECT_EQ(anchor->itemText(2), "Bottom");
    orient->setCurrentIndex(1); // horizontal
    EXPECT_EQ(anchor->itemText(0), "Left");
    EXPECT_EQ(anchor->itemText(2), "Right");
    orient->setCurrentIndex(0);
    EXPECT_EQ(anchor->itemText(0), "Top") << "switching back left the horizontal wording";
}

TEST(RefLinesDlg, ChangingTheOrientationDoesNotDisturbTheAnchor)
{
    RefLinesDialog d({}, RefLineStyle{});
    d.addLine();
    ctl<QComboBox>(d, "anchor0")->setCurrentIndex(1); // Center
    ctl<QComboBox>(d, "orient0")->setCurrentIndex(1); // horizontal

    const QList<RefLine> out = d.lines();
    ASSERT_EQ(out.size(), 1);
    EXPECT_EQ(out.at(0).anchor, RefAnchor::Center) << "relabelling the items moved the selection";
    EXPECT_EQ(out.at(0).orient, RefOrient::Horizontal);
}

TEST(RefLinesDlg, EveryAnchorSurvivesTheTrip)
{
    for (auto a : {RefAnchor::Start, RefAnchor::Center, RefAnchor::End}) {
        QList<RefLine> in;
        in << line(RefOrient::Vertical, 0.0, "x", QColor(1, 2, 3), a);
        RefLinesDialog d(in, RefLineStyle{});
        ASSERT_EQ(d.lines().size(), 1);
        EXPECT_EQ(d.lines().at(0).anchor, a) << "anchor " << static_cast<int>(a);
    }
}

TEST(RefLinesDlg, AnInvalidColourBecomesTheDefaultGrey)
{
    QList<RefLine> in;
    in << line(RefOrient::Vertical, 0.0, "x", QColor(), RefAnchor::Start);
    RefLinesDialog d(in, RefLineStyle{});
    ASSERT_EQ(d.lines().size(), 1);
    EXPECT_TRUE(d.lines().at(0).color.isValid()) << "an invalid colour was handed back";
}

TEST(RefLinesDlg, LabelsAreTrimmed)
{
    QList<RefLine> in;
    in << line(RefOrient::Vertical, 0.0, "  padded  ", QColor(1, 2, 3), RefAnchor::Start);
    RefLinesDialog d(in, RefLineStyle{});
    EXPECT_EQ(d.lines().at(0).label, "padded");
}

TEST(RefLinesDlg, TheLabelStyleControlsAreBounded)
{
    RefLineStyle in;
    in.fontSize = 1000.0;
    in.gap      = -50.0;
    RefLinesDialog d({}, in);
    const RefLineStyle out = d.labelStyle();
    EXPECT_LE(out.fontSize, 30.0) << "a font size past the maximum was accepted";
    EXPECT_GE(out.gap, 0.0) << "a negative gap was accepted";
}

/* ==================================================================== */
/*  The overlay palette                                                  */
/* ==================================================================== */

TEST(OverlayPalette, GivesAdjacentSeriesDifferentColours)
{
    for (int i = 0; i < 4; ++i)
        EXPECT_NE(overlaySeriesColor(i), overlaySeriesColor(i + 1)) << "series " << i;
}

TEST(OverlayPalette, WrapsRatherThanRunningOut)
{
    EXPECT_EQ(overlaySeriesColor(0), overlaySeriesColor(5));
    EXPECT_EQ(overlaySeriesColor(3), overlaySeriesColor(8));
    EXPECT_TRUE(overlaySeriesColor(1000).isValid());
}

TEST(OverlayPalette, ANegativeIndexStillGivesAColour)
{
    // overlaySeriesCount() cannot go negative today, but a modulo of a negative
    // number in C++ is negative and would index off the front of the palette
    EXPECT_TRUE(overlaySeriesColor(-1).isValid());
    EXPECT_TRUE(overlaySeriesColor(-7).isValid());
}

TEST(OverlayPalette, AvoidsTheRawAndProcessedSeriesColours)
{
    // an overlaid file that comes out the same colour as the chart's own data
    // looks like part of it
    const QColor rawDefault(100, 150, 255);
    const QColor procDefault(255, 125, 125);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NE(overlaySeriesColor(i), rawDefault) << "overlay " << i;
        EXPECT_NE(overlaySeriesColor(i), procDefault) << "overlay " << i;
    }
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
