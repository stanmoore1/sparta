// Unit tests for the displayed-image transform (src/displaytransform.cpp).
//
// This transform is applied in three places -- the pixels on screen, the
// ffmpeg movie export and the ImageMagick movie export -- and used to be
// written out three times. The copies had drifted: zoom out scaled by 0.9
// against a zoom in of 1.1, so zooming in and back out lost one percent per
// cycle and never returned to where it started. That bug was invisible to a
// test that only asked whether the picture changed.
//
// So the assertions here are round trips and inverses rather than "the result
// differs from the input": in and out must cancel, four quarter turns must be
// the identity, a mirror must be its own inverse, and clockwise must not equal
// counter-clockwise. Those distinguish a correct implementation from one that
// merely does something.

#include "displaytransform.h"

#include <QImage>
#include <QSize>

#include "gtest/gtest.h"

#include <string>

namespace {

std::string joined(const QStringList &list)
{
    return list.join(' ').toStdString();
}

// An asymmetric test image: symmetric content would make a mirror
// indistinguishable from the identity and hide a dead handler.
QImage sample()
{
    QImage img(8, 4, QImage::Format_RGB32);
    img.fill(Qt::black);
    img.setPixel(0, 0, qRgb(255, 0, 0));   // top left only
    img.setPixel(7, 0, qRgb(0, 255, 0));   // top right only
    img.setPixel(0, 3, qRgb(0, 0, 255));   // bottom left only
    return img;
}

// ---------------------------------------------------------------- zoom

TEST(DisplayTransform, ZoomOutIsTheExactInverseOfZoomIn)
{
    DisplayTransform t;
    t.zoomIn();
    t.zoomOut();
    EXPECT_DOUBLE_EQ(t.scale, 1.0);

    // and over several cycles, which is where a 0.9/1.1 pair drifts visibly
    for (int i = 0; i < 10; ++i)
        t.zoomIn();
    for (int i = 0; i < 10; ++i)
        t.zoomOut();
    EXPECT_NEAR(t.scale, 1.0, 1.0e-12);
}

TEST(DisplayTransform, ZoomInIsTenPercentAsAdvertised)
{
    DisplayTransform t;
    t.zoomIn();
    EXPECT_DOUBLE_EQ(t.scale, 1.1);
    t.zoomIn();
    EXPECT_DOUBLE_EQ(t.scale, 1.1 * 1.1);
}

TEST(DisplayTransform, ZoomOutStopsBeforeTheImageDisappears)
{
    DisplayTransform t;
    for (int i = 0; i < 100; ++i)
        t.zoomOut();
    EXPECT_DOUBLE_EQ(t.scale, DisplayTransform::MIN_SCALE);
}

// ------------------------------------------------------------- rotation

TEST(DisplayTransform, FourQuarterTurnsAreTheIdentity)
{
    DisplayTransform t;
    for (int i = 0; i < 4; ++i)
        t.rotateCw();
    EXPECT_EQ(t.rotation, 0);
    EXPECT_TRUE(t.isIdentity());
}

TEST(DisplayTransform, ClockwiseAndCounterClockwiseAreOpposites)
{
    DisplayTransform cw, ccw;
    cw.rotateCw();
    ccw.rotateCcw();
    EXPECT_EQ(cw.rotation, 90);
    EXPECT_EQ(ccw.rotation, 270);
    EXPECT_NE(cw, ccw);   // the two buttons must not be the same operation

    cw.rotateCcw();
    EXPECT_TRUE(cw.isIdentity());
}

TEST(DisplayTransform, QuarterTurnsExchangeWidthAndHeight)
{
    DisplayTransform t;
    EXPECT_EQ(transformedSize(QSize(8, 4), t), QSize(8, 4));
    t.rotateCw();
    EXPECT_EQ(transformedSize(QSize(8, 4), t), QSize(4, 8));
    t.rotateCw();
    EXPECT_EQ(transformedSize(QSize(8, 4), t), QSize(8, 4));
}

TEST(DisplayTransform, SizeCarriesRotationAndScaleTogether)
{
    DisplayTransform t;
    t.rotateCw();
    t.scale = 2.0;
    EXPECT_EQ(transformedSize(QSize(8, 4), t), QSize(8, 16));
}

// --------------------------------------------------------------- mirrors

TEST(DisplayTransform, EachMirrorIsItsOwnInverse)
{
    DisplayTransform t;
    t.mirrorH();
    t.mirrorH();
    EXPECT_TRUE(t.isIdentity());
    t.mirrorV();
    t.mirrorV();
    EXPECT_TRUE(t.isIdentity());
}

TEST(DisplayTransform, TheTwoMirrorsAreDifferentOperations)
{
    DisplayTransform h, v;
    h.mirrorH();
    v.mirrorV();
    EXPECT_NE(h, v);
}

// ------------------------------------------------------ applied to pixels

TEST(ApplyDisplayTransform, IdentityReturnsTheImageUnchanged)
{
    const QImage src = sample();
    EXPECT_EQ(applyDisplayTransform(src, DisplayTransform()), src);
}

TEST(ApplyDisplayTransform, MirroringHorizontallyTwiceRestoresTheImage)
{
    const QImage src = sample();
    DisplayTransform t;
    t.mirrorH();
    const QImage once = applyDisplayTransform(src, t);
    EXPECT_NE(once, src);
    t.mirrorH();
    EXPECT_EQ(applyDisplayTransform(src, t), src);
}

TEST(ApplyDisplayTransform, HorizontalMirrorMovesTheCornerAcross)
{
    DisplayTransform t;
    t.mirrorH();
    const QImage out = applyDisplayTransform(sample(), t);
    // the red pixel started at the top left and must end at the top right
    EXPECT_EQ(out.pixel(7, 0), qRgb(255, 0, 0));
    EXPECT_EQ(out.pixel(0, 0), qRgb(0, 255, 0));
}

TEST(ApplyDisplayTransform, VerticalMirrorMovesTheCornerDown)
{
    DisplayTransform t;
    t.mirrorV();
    const QImage out = applyDisplayTransform(sample(), t);
    EXPECT_EQ(out.pixel(0, 3), qRgb(255, 0, 0));
    EXPECT_EQ(out.pixel(0, 0), qRgb(0, 0, 255));
}

TEST(ApplyDisplayTransform, ClockwiseRotationLandsTheCornerOnTheRight)
{
    DisplayTransform t;
    t.rotateCw();
    const QImage out = applyDisplayTransform(sample(), t);
    ASSERT_EQ(out.size(), QSize(4, 8));
    // a clockwise quarter turn sends the top-left corner to the top right
    EXPECT_EQ(out.pixel(3, 0), qRgb(255, 0, 0));
}

TEST(ApplyDisplayTransform, CounterClockwiseIsNotClockwise)
{
    DisplayTransform cw, ccw;
    cw.rotateCw();
    ccw.rotateCcw();
    const QImage src = sample();
    EXPECT_NE(applyDisplayTransform(src, cw), applyDisplayTransform(src, ccw));
}

TEST(ApplyDisplayTransform, FourRotationsReturnTheOriginalPixels)
{
    const QImage src = sample();
    QImage out       = src;
    DisplayTransform quarter;
    quarter.rotateCw();
    for (int i = 0; i < 4; ++i)
        out = applyDisplayTransform(out, quarter);
    EXPECT_EQ(out, src);
}

// ------------------------------------------------ movie export arguments

TEST(FfmpegFilterArgs, IdentityNeedsNoFilter)
{
    EXPECT_TRUE(ffmpegFilterArgs(DisplayTransform()).isEmpty());
}

TEST(FfmpegFilterArgs, RotationsMapToTransposeInTheRightDirection)
{
    DisplayTransform t;
    t.rotateCw();
    EXPECT_EQ(joined(ffmpegFilterArgs(t)), "-vf transpose=1");
    t.rotateCw();
    EXPECT_EQ(joined(ffmpegFilterArgs(t)), "-vf transpose=1,transpose=1");
    t.rotateCw();
    EXPECT_EQ(joined(ffmpegFilterArgs(t)), "-vf transpose=2");
}

TEST(FfmpegFilterArgs, MirrorsMapToTheMatchingFlip)
{
    DisplayTransform h;
    h.mirrorH();
    EXPECT_EQ(joined(ffmpegFilterArgs(h)), "-vf hflip");

    DisplayTransform v;
    v.mirrorV();
    EXPECT_EQ(joined(ffmpegFilterArgs(v)), "-vf vflip");
}

TEST(FfmpegFilterArgs, FiltersAreChainedWithCommasInOrder)
{
    DisplayTransform t;
    t.scale = 2.0;
    t.rotateCw();
    t.mirrorH();
    t.mirrorV();
    EXPECT_EQ(joined(ffmpegFilterArgs(t)), "-vf scale=iw*2:-1,transpose=1,hflip,vflip");
}

TEST(MagickTransformArgs, IdentityNeedsNoArguments)
{
    EXPECT_TRUE(magickTransformArgs(DisplayTransform()).isEmpty());
}

TEST(MagickTransformArgs, EachPartMapsToItsOwnOption)
{
    DisplayTransform t;
    t.scale = 0.5;
    t.rotateCw();
    t.mirrorH();
    t.mirrorV();
    EXPECT_EQ(joined(magickTransformArgs(t)), "-resize 50% -rotate 90 -flop -flip");
}

TEST(MagickTransformArgs, MirrorsUseTheOppositeOptionNames)
{
    // -flop is the horizontal mirror and -flip the vertical one, which is the
    // reverse of what the names suggest and an easy place to swap them
    DisplayTransform h;
    h.mirrorH();
    EXPECT_EQ(joined(magickTransformArgs(h)), "-flop");

    DisplayTransform v;
    v.mirrorV();
    EXPECT_EQ(joined(magickTransformArgs(v)), "-flip");
}

// The screen and the two exporters must agree about orientation, or a movie
// comes out turned the wrong way from the preview it was exported from.
TEST(MovieExport, BothExportersAgreeWithTheScreenAboutDirection)
{
    DisplayTransform t;
    t.rotateCw();
    EXPECT_EQ(transformedSize(QSize(8, 4), t), QSize(4, 8));
    EXPECT_EQ(joined(ffmpegFilterArgs(t)), "-vf transpose=1");   // clockwise
    EXPECT_EQ(joined(magickTransformArgs(t)), "-rotate 90");     // clockwise
}

} // namespace
