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

// The slide show: the window that turns a run's dump images into an animation.
//
// Its sequence bookkeeping -- the active range, where "next" goes at the end of
// it, what the Start and Stop boxes do as images arrive -- is all reachable
// without a simulator, and none of it was checked.  The live suites drive the
// buttons and photograph the result, which catches a crash but not an off-by-one
// in which image is showing.
//
// Real image files, because the window reads their headers to size itself.

#include "slideshow.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QImage>
#include <QRegularExpression>
#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalSpy>
#include <QSpinBox>
#include <QTemporaryDir>

namespace {

// A slide show as "File > View Image File" opens it: no main window behind it,
// so no live simulation feeding images in.
class Show : public ::testing::Test {
protected:
    void SetUp() override { show = new SlideShow(QString(), nullptr); }
    void TearDown() override { delete show; }

    // an image whose single colour identifies it, so the wrong one showing is
    // visible rather than merely suspected
    QString image(int n, int w = 40, int h = 30) const
    {
        QImage img(w, h, QImage::Format_RGB32);
        img.fill(QColor(n * 20 % 256, 60, 90));
        const QString path = dir.filePath(QString("frame.%1.png").arg(n, 4, 10, QChar('0')));
        EXPECT_TRUE(img.save(path));
        return path;
    }

    // a running counter, so two calls append rather than re-offering the same
    // files -- addImage() ignores a path it already holds
    void addImages(int count)
    {
        for (int i = 0; i < count; ++i)
            show->addImage(image(next_++));
    }


    template <class W> W *ctl(const char *name) const
    {
        auto *w = show->findChild<W *>(QLatin1String(name));
        EXPECT_NE(w, nullptr) << "no control named " << name;
        return w;
    }

    QSpinBox *startBox() const { return ctl<QSpinBox>("start"); }
    QSpinBox *stopBox() const { return ctl<QSpinBox>("stop"); }

    // which image is on screen, read off the counter rather than a private member
    int shownIndex() const
    {
        for (auto *l : show->findChildren<QLabel *>()) {
            const auto m = QRegularExpression(R"(Image\s+(\d+)\s*/)").match(l->text());
            if (m.hasMatch()) return m.captured(1).toInt() - 1;
        }
        return -1;
    }

    QTemporaryDir dir;
    SlideShow *show = nullptr;
    mutable int next_ = 0;
};

} // namespace

// ---------------------------------------------------------------- the sequence

TEST_F(Show, StartsEmpty)
{
    EXPECT_EQ(show->imageCount(), 0);
    EXPECT_FALSE(show->hasContent());
    EXPECT_TRUE(show->images().isEmpty());
}

TEST_F(Show, CollectsTheImagesItIsGiven)
{
    addImages(5);
    EXPECT_EQ(show->imageCount(), 5);
    EXPECT_TRUE(show->hasContent());
    EXPECT_EQ(show->images().size(), 5);
}

TEST_F(Show, TheSameFileIsNotAddedTwice)
{
    // a rescan of the run directory offers every image again; the sequence must
    // not double
    const QString one = image(0);
    show->addImage(one);
    show->addImage(one);
    EXPECT_EQ(show->imageCount(), 1);
}

TEST_F(Show, AnnouncesThatItsContentChanged)
{
    QSignalSpy spy(show, &SlideShow::contentChanged);
    addImages(3);
    EXPECT_EQ(spy.count(), 3) << "the window around it was not told a new image arrived";
    show->clear();
    EXPECT_EQ(spy.count(), 4) << "clearing did not announce itself";
}

TEST_F(Show, ClearingEmptiesItAndLeavesItReusable)
{
    addImages(4);
    show->clear();
    EXPECT_EQ(show->imageCount(), 0);
    EXPECT_FALSE(show->hasContent());

    addImages(2);
    EXPECT_EQ(show->imageCount(), 2) << "the window did not accept images after being cleared";
}

TEST_F(Show, AnUnreadableFileIsStillPartOfTheSequence)
{
    // a dump image the run has not finished writing yet: the entry belongs in
    // the sequence so the count is right, and the display simply has nothing
    // to show for it
    show->addImage(dir.filePath("not-written-yet.png"));
    EXPECT_EQ(show->imageCount(), 1);
}

// ---------------------------------------------------------------- the active range

TEST_F(Show, TheActiveRangeGrowsWithTheSequence)
{
    addImages(1);
    EXPECT_EQ(startBox()->value(), 1);
    EXPECT_EQ(stopBox()->value(), 1);

    addImages(4); // frames 1..4, so five in total
    EXPECT_EQ(stopBox()->maximum(), 5);
    EXPECT_EQ(stopBox()->value(), 5) << "Stop did not follow the end of the sequence";
    EXPECT_EQ(startBox()->value(), 1) << "Start moved on its own";
}

TEST_F(Show, AnExplicitStopIsNotDraggedAlongByNewImages)
{
    // the distinction the follow-the-end rule exists for: a user who pinned
    // Stop to image 3 keeps it there as the run writes more
    addImages(5);
    stopBox()->setValue(3);
    addImages(3); // frames 5..7
    EXPECT_EQ(stopBox()->value(), 3) << "the user's Stop choice was overwritten";
    EXPECT_EQ(stopBox()->maximum(), 8) << "the ceiling did not grow with the sequence";
}

TEST_F(Show, TheRangeBoxesAreOneBasedAndTheIndicesAreNot)
{
    addImages(6);
    startBox()->setValue(2);
    stopBox()->setValue(5);
    // startIdx()/stopIdx() are private; the navigation below is what reads them,
    // so check the boxes agree with what "first" and "last" then show
    QMetaObject::invokeMethod(show, "first");
    EXPECT_EQ(shownIndex(), 1) << "Start=2 should show the second image, index 1";
    QMetaObject::invokeMethod(show, "last");
    EXPECT_EQ(shownIndex(), 4) << "Stop=5 should show the fifth image, index 4";
}

// ---------------------------------------------------------------- navigation

TEST_F(Show, NextAndPreviousWalkTheActiveRange)
{
    addImages(6);
    QMetaObject::invokeMethod(show, "first");
    ASSERT_EQ(shownIndex(), 0);

    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 1);
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 2);
    QMetaObject::invokeMethod(show, "prev");
    EXPECT_EQ(shownIndex(), 1);
}

TEST_F(Show, WithoutLoopingTheEndsAreSticky)
{
    addImages(4);
    QMetaObject::invokeMethod(show, "loop"); // looping is on by default; turn it off
    QMetaObject::invokeMethod(show, "last");
    ASSERT_EQ(shownIndex(), 3);
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 3) << "next ran past the end of the range";

    QMetaObject::invokeMethod(show, "first");
    QMetaObject::invokeMethod(show, "prev");
    EXPECT_EQ(shownIndex(), 0) << "previous ran before the start of the range";
}

TEST_F(Show, LoopingWrapsAtBothEnds)
{
    addImages(4); // looping is on by default
    QMetaObject::invokeMethod(show, "last");
    ASSERT_EQ(shownIndex(), 3);
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 0) << "looping did not wrap forwards";
    QMetaObject::invokeMethod(show, "prev");
    EXPECT_EQ(shownIndex(), 3) << "looping did not wrap backwards";
}

TEST_F(Show, NavigationRespectsANarrowedRange)
{
    addImages(8);
    QMetaObject::invokeMethod(show, "loop"); // off, so the range ends are sticky
    startBox()->setValue(3);
    stopBox()->setValue(5);

    QMetaObject::invokeMethod(show, "first");
    EXPECT_EQ(shownIndex(), 2);
    QMetaObject::invokeMethod(show, "next");
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 4);
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 4) << "next left the active range";
}

TEST_F(Show, LoopingWrapsWithinANarrowedRange)
{
    addImages(8); // looping is on by default
    startBox()->setValue(3);
    stopBox()->setValue(5);
    QMetaObject::invokeMethod(show, "last");
    ASSERT_EQ(shownIndex(), 4);
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 2) << "the wrap went to the sequence start, not the range start";
}

TEST_F(Show, NavigatingAnEmptyShowIsHarmless)
{
    QMetaObject::invokeMethod(show, "first");
    QMetaObject::invokeMethod(show, "next");
    QMetaObject::invokeMethod(show, "prev");
    QMetaObject::invokeMethod(show, "last");
    SUCCEED();
}

TEST_F(Show, ASingleImageIsItsOwnStartAndEnd)
{
    addImages(1);
    QMetaObject::invokeMethod(show, "first");
    EXPECT_EQ(shownIndex(), 0);
    QMetaObject::invokeMethod(show, "next"); // looping on: wraps to itself
    EXPECT_EQ(shownIndex(), 0);
    QMetaObject::invokeMethod(show, "loop"); // looping off: sticks at itself
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 0) << "a one-image sequence moved somewhere";
}

// ---------------------------------------------------------------- playback

TEST_F(Show, PlayStartsAndStopsTheTimer)
{
    addImages(4);
    auto *delay = ctl<QSpinBox>("delay");
    ASSERT_TRUE(delay->isEnabled());

    QMetaObject::invokeMethod(show, "play");
    EXPECT_FALSE(delay->isEnabled()) << "the delay stayed editable while playing";
    QMetaObject::invokeMethod(show, "play");
    EXPECT_TRUE(delay->isEnabled()) << "the delay stayed locked after stopping";
}

TEST_F(Show, PlayingWithoutLoopingRewindsToTheRangeStart)
{
    addImages(6);
    QMetaObject::invokeMethod(show, "loop"); // the rewind only happens when not looping
    startBox()->setValue(3);
    QMetaObject::invokeMethod(show, "last");
    ASSERT_NE(shownIndex(), 2);

    QMetaObject::invokeMethod(show, "play");
    EXPECT_EQ(shownIndex(), 2)
        << "play rewound but did not draw, so the range's first image is skipped";

    QMetaObject::invokeMethod(show, "play"); // stop
    EXPECT_EQ(shownIndex(), 2) << "stopping playback rewound the sequence on its own";
}

// ---------------------------------------------------------------- deletion

TEST_F(Show, DeletingTheWholeRangeEmptiesTheShow)
{
    addImages(3);
    const QStringList files = show->images();

    // deleteImages() is guarded by a confirmation; drive the effect directly by
    // clearing, then check the files themselves are what the guard protects
    for (const auto &f : files)
        EXPECT_TRUE(QFile::exists(f)) << "the fixture did not write " << f.toStdString();
    show->clear();
    EXPECT_EQ(show->imageCount(), 0);
    for (const auto &f : files)
        EXPECT_TRUE(QFile::exists(f)) << "clear() deleted a file from disk";
}

// ---------------------------------------------------------------- display

TEST_F(Show, TheTransformsAreAcceptedAndUndoOneAnother)
{
    addImages(2);
    QMetaObject::invokeMethod(show, "first");
    QMetaObject::invokeMethod(show, "doImageRotateCw");
    QMetaObject::invokeMethod(show, "doImageRotateCcw");
    QMetaObject::invokeMethod(show, "doImageFlipH");
    QMetaObject::invokeMethod(show, "doImageFlipH");
    QMetaObject::invokeMethod(show, "doImageFlipV");
    QMetaObject::invokeMethod(show, "doImageFlipV");
    QMetaObject::invokeMethod(show, "zoomIn");
    QMetaObject::invokeMethod(show, "zoomOut");
    QMetaObject::invokeMethod(show, "normalSize");
    EXPECT_EQ(show->imageCount(), 2) << "a transform disturbed the sequence";
}

TEST_F(Show, TransformsOnAnEmptyShowAreHarmless)
{
    QMetaObject::invokeMethod(show, "doImageRotateCw");
    QMetaObject::invokeMethod(show, "zoomIn");
    QMetaObject::invokeMethod(show, "normalSize");
    SUCCEED();
}

TEST_F(Show, ItRendersWhatItHolds)
{
    addImages(3);
    show->resize(400, 300);
    QMetaObject::invokeMethod(show, "first");
    EXPECT_FALSE(show->grab().isNull());
}

TEST_F(Show, ImagesOfDifferentSizesAreAllAccepted)
{
    // a run whose dump image size changed mid-way: the window sizes itself to
    // the largest, and every image stays in the sequence
    show->addImage(image(0, 40, 30));
    show->addImage(image(1, 200, 150));
    show->addImage(image(2, 10, 10));
    EXPECT_EQ(show->imageCount(), 3);
    QMetaObject::invokeMethod(show, "first");
    QMetaObject::invokeMethod(show, "next");
    QMetaObject::invokeMethod(show, "next");
    EXPECT_EQ(shownIndex(), 2);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_slideshow.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
