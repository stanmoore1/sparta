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

// The chart renderer.
//
// PlotWidget draws every chart the application shows and had no test of any
// kind. Its inputs are a handful of setters and a list of non-owned series;
// what it produces is pixels, so that is what this checks: the widget is
// rendered into an image and the image is asked whether the thing that was
// supposed to appear actually did.
//
// Rendering rather than reading back state is the point. A setter that stores
// its argument and a renderer that ignores it look identical from the outside,
// and that is exactly the defect this class could hide -- there is nothing else
// watching it.

#include <gtest/gtest.h>

#include <QApplication>
#include <QColor>
#include <QFont>
#include <QImage>
#include <QPainter>
#include <QSet>

#include <memory>

#include "plotseries.h"
#include "plotwidget.h"

namespace {

// Render the widget offscreen and hand back the pixels.
QImage shoot(PlotWidget &w, int width = 400, int height = 300)
{
    w.resize(width, height);
    QImage img(width, height, QImage::Format_RGB32);
    img.fill(Qt::white);
    w.render(&img);
    return img;
}

int countOf(const QImage &img, const QColor &c)
{
    const QRgb want = c.rgb();
    int n           = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x)
            if ((img.pixel(x, y) & 0x00ffffff) == (want & 0x00ffffff)) ++n;
    return n;
}

size_t distinctColors(const QImage &img)
{
    QSet<QRgb> seen;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x)
            seen.insert(img.pixel(x, y) & 0x00ffffff);
    return size_t(seen.size());
}

int differing(const QImage &a, const QImage &b)
{
    if (a.size() != b.size()) return -1;
    int n = 0;
    for (int y = 0; y < a.height(); ++y)
        for (int x = 0; x < a.width(); ++x)
            if (a.pixel(x, y) != b.pixel(x, y)) ++n;
    return n;
}

// A diagonal line in a distinctive colour, so it can be counted in the picture.
std::unique_ptr<PlotSeries> ramp(const QColor &color = QColor(255, 0, 0), int n = 50)
{
    auto s   = std::make_unique<PlotSeries>();
    s->color = color;
    s->width = 2.0;
    s->name  = "ramp";
    for (int i = 0; i < n; ++i)
        s->append(double(i) / (n - 1), double(i) / (n - 1));
    return s;
}

} // namespace

// -------------------------------------------------------------- registration

TEST(PlotWidget, StartsWithNoSeries)
{
    PlotWidget w;
    auto s = ramp();
    EXPECT_FALSE(w.hasSeries(s.get()));
}

TEST(PlotWidget, RegistersAndUnregistersWithoutOwning)
{
    PlotWidget w;
    auto s = ramp();

    w.addSeries(s.get());
    EXPECT_TRUE(w.hasSeries(s.get()));

    // adding twice must not register it twice, or every point is drawn twice
    w.addSeries(s.get());
    EXPECT_TRUE(w.hasSeries(s.get()));
    w.removeSeries(s.get());
    EXPECT_FALSE(w.hasSeries(s.get()))
        << "the series was registered more than once, so removing it left a copy behind";

    // and the caller still owns it: this must not be a double free
    w.addSeries(s.get());
    w.clearSeries();
    EXPECT_FALSE(w.hasSeries(s.get()));
    EXPECT_EQ(s->count(), 50) << "clearSeries() disturbed a series it does not own";
}

TEST(PlotWidget, RemovingSomethingNeverAddedIsHarmless)
{
    PlotWidget w;
    auto a = ramp();
    auto b = ramp();
    w.addSeries(a.get());
    w.removeSeries(b.get());
    EXPECT_TRUE(w.hasSeries(a.get())) << "removing an unregistered series removed another one";
}

// ------------------------------------------------------------------ drawing

TEST(PlotWidget, AnEmptyChartIsStillAxesAndNotABlankField)
{
    PlotWidget w;
    const QImage img = shoot(w);
    EXPECT_GT(distinctColors(img), 1u)
        << "the chart rendered as a single flat colour: no axes, no frame, nothing";
}

TEST(PlotWidget, ARegisteredSeriesReachesThePicture)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);
    const QImage before = shoot(w);

    auto s = ramp(QColor(255, 0, 0));
    w.addSeries(s.get());
    const QImage after = shoot(w);

    EXPECT_GT(countOf(after, QColor(255, 0, 0)), 20)
        << "the series' own colour is not in the picture, so it was not drawn";
    EXPECT_GT(differing(before, after), 20);
}

TEST(PlotWidget, AnInvisibleSeriesIsNotDrawn)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);
    auto s = ramp(QColor(0, 128, 0));
    w.addSeries(s.get());

    const QImage shown = shoot(w);
    s->setVisible(false);
    const QImage hidden = shoot(w);

    EXPECT_GT(countOf(shown, QColor(0, 128, 0)), 20);
    EXPECT_EQ(countOf(hidden, QColor(0, 128, 0)), 0)
        << "a series marked invisible was drawn anyway";
}

TEST(PlotWidget, TheSeriesColourIsTheColourThatIsDrawn)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);
    auto s = ramp(QColor(0, 0, 255));
    w.addSeries(s.get());

    const QImage img = shoot(w);
    EXPECT_GT(countOf(img, QColor(0, 0, 255)), 20);
    EXPECT_EQ(countOf(img, QColor(255, 0, 0)), 0) << "a colour nobody asked for is in the picture";
}

TEST(PlotWidget, TheRangeDecidesWhatIsInsideThePlot)
{
    // The same series against two ranges: one that contains it and one far
    // above it. A renderer that ignores the axis range draws both alike.
    PlotWidget w;
    auto s = ramp(QColor(255, 0, 0));
    w.addSeries(s.get());

    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);
    const int inside = countOf(shoot(w), QColor(255, 0, 0));

    w.setYRange(100.0, 200.0);
    const int outside = countOf(shoot(w), QColor(255, 0, 0));

    EXPECT_GT(inside, 20);
    EXPECT_LT(outside, inside)
        << "moving the y range far above the data drew the same amount of it";
}

// ------------------------------------------------------------------ chrome

TEST(PlotWidget, TheTitleAndAxisTitlesAreDrawn)
{
    PlotWidget w;
    const QImage bare = shoot(w);

    w.setTitle("Stats: in.circle");
    const QImage titled = shoot(w);
    EXPECT_GT(differing(bare, titled), 20) << "setting the chart title changed nothing on screen";

    w.setXTitle("Time step");
    const QImage xt = shoot(w);
    EXPECT_GT(differing(titled, xt), 20) << "setting the x-axis title changed nothing on screen";

    w.setYTitle("CPU");
    const QImage yt = shoot(w);
    EXPECT_GT(differing(xt, yt), 20) << "setting the y-axis title changed nothing on screen";

    // and they read back, since the chart window asks for them
    EXPECT_EQ(w.xTitle(), QString("Time step"));
    EXPECT_EQ(w.yTitle(), QString("CPU"));
}

TEST(PlotWidget, TheGridlinesCanBeTurnedOff)
{
    PlotWidget w;
    w.setGrid(true, true);
    const QImage both = shoot(w);
    w.setGrid(true, false);
    const QImage majorOnly = shoot(w);
    w.setGrid(false, false);
    const QImage none = shoot(w);

    EXPECT_GT(differing(both, majorOnly), 10) << "turning the minor grid off changed nothing";
    EXPECT_GT(differing(majorOnly, none), 10) << "turning the major grid off changed nothing";
}

TEST(PlotWidget, TheLegendAppearsWhereItIsPlaced)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);
    auto s = ramp(QColor(255, 0, 0));
    s->name = "CPU";
    w.addSeries(s.get());

    w.setLegendPos(LegendPos::Off);
    const QImage off = shoot(w);
    w.setLegendPos(LegendPos::TopLeft);
    const QImage tl = shoot(w);
    w.setLegendPos(LegendPos::BottomRight);
    const QImage br = shoot(w);

    EXPECT_GT(differing(off, tl), 20) << "turning the legend on changed nothing on screen";
    EXPECT_GT(differing(tl, br), 20)
        << "the legend renders identically in two different corners, so the placement is ignored";
}

// The label format is not simply obeyed, and that is deliberate: the renderer
// derives the number of decimals from the tick spacing so that closely spaced
// ticks cannot collapse to identical labels. A caller's own format only
// survives when it asks for integers and the ticks are at least 1 apart.
TEST(PlotWidget, TheDecimalsFollowTheTickSpacingRatherThanTheGivenFormat)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setXLabelFormat("%g");
    const QImage g = shoot(w);
    w.setXLabelFormat("%.4f");
    const QImage f = shoot(w);
    EXPECT_EQ(differing(g, f), 0)
        << "a fractional format changed the labels; the spacing is supposed to decide, or "
           "ticks 0.05 apart would all read the same";
}

// The observable consequence: a fine range gets more decimals than a coarse
// one, whatever format the caller asked for.
//
// (effectiveFormat() has a branch that passes an integer format straight
// through when the ticks are at least 1 apart. It cannot be observed: whenever
// that branch is taken the spacing-derived format is "%.0f", which prints the
// same text as "%d". Nothing here depends on which of the two is used.)
TEST(PlotWidget, AFineRangeGetsMoreDecimalsThanACoarseOne)
{
    PlotWidget coarse;
    coarse.setXRange(0.0, 1000.0);
    const QImage far = shoot(coarse);

    PlotWidget fine;
    fine.setXRange(0.0, 0.001);
    const QImage near = shoot(fine);

    EXPECT_GT(differing(far, near), 20)
        << "a range a million times finer drew the identical tick labels, so the decimals do "
           "not follow the spacing and closely spaced ticks would all read the same";
}

// A reference line is a series drawn dashed with a label beside it; the chart
// window's Reference Lines dialog is the only thing that makes one, and it had
// nothing checking that its style arguments reach the picture.
TEST(PlotWidget, ReferenceLabelStyleReachesThePicture)
{
    PlotWidget w;
    w.setXRange(0.0, 1.0);
    w.setYRange(0.0, 1.0);

    PlotSeries ref;
    ref.isReference = true;
    ref.refLabel    = "mean";
    ref.color       = QColor(255, 0, 255);
    ref.style       = Qt::DashLine;
    ref.append(0.0, 0.5);
    ref.append(1.0, 0.5);
    w.addSeries(&ref);

    w.setRefLabelStyle(0.0, 4.0, false);
    const QImage plain = shoot(w);
    w.setRefLabelStyle(14.0, 12.0, true);
    const QImage styled = shoot(w);

    EXPECT_GT(countOf(plain, QColor(255, 0, 255)), 10) << "the reference line was not drawn";
    EXPECT_GT(differing(plain, styled), 20)
        << "the reference-label size, gap and box were stored and never used for drawing";

    w.clearSeries();   // ref lives on the stack; unregister before it goes away
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
