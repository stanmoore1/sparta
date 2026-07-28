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

// Importing a movie, against a real one: probeMovie() and extractMovieFrames().
//
// test_movieimport.cpp covers the pure half -- parsing ffprobe's JSON, the frame
// rate, the arithmetic of a range and a stride -- from hand-written strings.
// What had never run is the half that talks to ffmpeg: probing a container that
// does not store a frame count, turning a selection into a filter expression,
// and deciding afterwards whether the extraction worked.
//
// That matters because the frames are what the user then measures.  A stride
// that is off by one, or a range that starts a frame early, produces a slide
// show that looks entirely normal and is not the part of the trajectory that
// was asked for -- there is nothing in the images to say so.
//
// So the fixture builds movies whose frames are individually identifiable: each
// frame is a flat colour derived from its number, and every check reads the
// colours back out of the extracted PNGs.  ffmpeg does the work both ways, so a
// disagreement is in the code under test rather than in the codec.
//
// One thing mutation testing turned up that is worth writing down: the upper
// bound of the range is enforced twice, once by the select filter and again by
// "-frames:v <count>".  Breaking either alone changes nothing observable,
// because the other still stops the extraction in the right place.  That is
// belt and braces rather than a gap in these tests -- the lower bound and the
// stride, which are enforced once, are both pinned below.

#include "movieimport.h"

#include "constants.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QColor>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QProcess>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>

namespace {

bool haveFfmpeg()
{
    QProcess p;
    p.start("ffmpeg", {"-version"});
    if (!p.waitForFinished(10000) || p.exitCode() != 0) return false;
    QProcess q;
    q.start("ffprobe", {"-version"});
    return q.waitForFinished(10000) && q.exitCode() == 0;
}

#define REQUIRE_FFMPEG() \
    if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg and ffprobe are needed to import a movie"

/// Dismisses the progress dialog's siblings; the extraction shows a modal
/// progress dialog that nothing here should have to answer.
class Modals : public QObject {
public:
    explicit Modals(int budgetMs = 60000) : left(budgetMs)
    {
        timer.setInterval(20);
        connect(&timer, &QTimer::timeout, this, &Modals::poll);
        timer.start();
    }

private:
    void poll()
    {
        if ((left -= 20) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(QApplication::activeModalWidget())) d->reject();
        }
    }
    QTimer timer;
    int left;
};

class MovieLive : public ::testing::Test {
protected:
    /// A movie of @p frames flat-coloured frames, one colour per frame, so an
    /// extracted PNG says which frame of the original it came from.  The green
    /// channel carries the frame number.
    ///
    /// @param container the file extension, which selects the container: mp4
    /// stores a frame count, webm does not.
    QString makeMovie(int frames, const QString &container = "mp4", int fps = 10)
    {
        const QString src = dir.filePath("src");
        QDir().mkpath(src);
        for (int i = 0; i < frames; ++i) {
            QImage img(64, 48, QImage::Format_RGB32);
            img.fill(QColor(20, colourFor(i), 40));
            if (!img.save(QString("%1/in_%2.png").arg(src).arg(i, 5, 10, QChar('0'))))
                return {};
        }
        const QString out = dir.filePath("movie." + container);
        QProcess p;
        // lossless, no inter-frame prediction, one frame in = one frame out, so
        // the colours survive the round trip exactly
        QStringList args{"-y", "-nostdin", "-loglevel", "error", "-framerate",
                         QString::number(fps), "-i",     QString("%1/in_%05d.png").arg(src)};
        if (container == "webm")
            args << "-c:v" << "libvpx-vp9" << "-lossless" << "1";
        else
            args << "-c:v" << "libx264" << "-qp" << "0" << "-pix_fmt" << "yuv444p"
                 << "-g" << "1";
        args << "-vsync" << "0" << out;
        p.start("ffmpeg", args);
        if (!p.waitForFinished(120000) || p.exitCode() != 0) return {};
        return QFile::exists(out) ? out : QString();
    }

    /// The colour a frame is painted, spaced so neighbouring frames are far
    /// apart and a one-frame error cannot hide in codec noise.
    static int colourFor(int frame) { return 10 + 20 * frame; }

    /// Which source frame an extracted PNG came from, read back from its colour.
    static int frameOf(const QString &png)
    {
        const QImage img(png);
        if (img.isNull()) return -1;
        const int green = QColor(img.pixel(img.width() / 2, img.height() / 2)).green();
        const int n     = qRound((green - 10) / 20.0);
        // reject a colour that is not close to any frame's: that would mean the
        // round trip lost the encoding rather than the frame being misidentified
        return std::abs(colourFor(n) - green) <= 4 ? n : -1;
    }

    static QList<int> framesOf(const QStringList &pngs)
    {
        QList<int> out;
        for (const auto &p : pngs) out << frameOf(p);
        return out;
    }

    QString outDir(const QString &name = "out")
    {
        const QString p = dir.filePath(name);
        QDir().mkpath(p);
        return p;
    }

    QTemporaryDir dir;
};

} // namespace

// ------------------------------------------------------------------- probing

TEST_F(MovieLive, ProbesTheSizeRateAndLengthOfARealMovie)
{
    REQUIRE_FFMPEG();
    const QString movie = makeMovie(12);
    ASSERT_FALSE(movie.isEmpty()) << "could not build a test movie";

    const MovieInfo info = probeMovie(movie);
    ASSERT_TRUE(info.valid) << info.error.toStdString();
    EXPECT_EQ(info.width, 64);
    EXPECT_EQ(info.height, 48);
    EXPECT_EQ(info.frames, 12) << "the frame count is not the number of frames encoded";
    EXPECT_NEAR(info.fps, 10.0, 1e-6);
    EXPECT_NEAR(info.duration, 1.2, 0.25) << "the duration does not match 12 frames at 10 fps";
}

TEST_F(MovieLive, CountsTheFramesOfAContainerThatDoesNotStoreThem)
{
    // webm keeps no frame count, so probeMovie falls back to counting video
    // packets.  Without that fallback the import refuses a perfectly good file.
    REQUIRE_FFMPEG();
    const QString movie = makeMovie(9, "webm");
    if (movie.isEmpty()) GTEST_SKIP() << "this ffmpeg cannot write webm";

    const MovieInfo info = probeMovie(movie);
    ASSERT_TRUE(info.valid) << info.error.toStdString();
    EXPECT_EQ(info.frames, 9) << "the packet-counting fallback did not find every frame";
}

TEST_F(MovieLive, AFileThatIsNotAMovieIsRefusedWithWhatFfprobeSaid)
{
    REQUIRE_FFMPEG();
    const QString bogus = dir.filePath("notamovie.mp4");
    QFile f(bogus);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("this is not a movie, whatever the extension says\n");
    f.close();

    const MovieInfo info = probeMovie(bogus);
    EXPECT_FALSE(info.valid) << "a text file was accepted as a movie";
    EXPECT_FALSE(info.error.isEmpty()) << "it was refused without saying why";
}

TEST_F(MovieLive, AMissingFileIsRefused)
{
    REQUIRE_FFMPEG();
    const MovieInfo info = probeMovie(dir.filePath("never-written.mp4"));
    EXPECT_FALSE(info.valid);
    EXPECT_FALSE(info.error.isEmpty());
}

// ---------------------------------------------------------------- extracting

TEST_F(MovieLive, ExtractsExactlyTheFramesThatWereAskedFor)
{
    // the whole point: the frames handed to the slide show are the ones the
    // user selected, in order.  Nothing in the images themselves would say
    // otherwise, so the colours are checked rather than the count.
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(10);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 3, 6, 1, err);
    ASSERT_TRUE(err.isEmpty()) << err.toStdString();
    ASSERT_EQ(frames.size(), 4) << "frames 3..6 is four frames";

    // the dialog counts from 1 and ffmpeg from 0, so frame 3 is source frame 2
    EXPECT_EQ(framesOf(frames), (QList<int>{2, 3, 4, 5}))
        << "the extracted frames are not the ones selected";
}

TEST_F(MovieLive, AStrideTakesEveryNthFrameFromTheStartOfTheRange)
{
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(12);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 2, 10, 3, err);
    ASSERT_TRUE(err.isEmpty()) << err.toStdString();

    // frames 2, 5, 8 counted from 1 -> source frames 1, 4, 7
    EXPECT_EQ(frames.size(), selectedFrameCount(2, 10, 3));
    EXPECT_EQ(framesOf(frames), (QList<int>{1, 4, 7}))
        << "the stride did not start at the first frame of the range";
}

TEST_F(MovieLive, TheWholeMovieCanBeExtracted)
{
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(8);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 1, 8, 1, err);
    ASSERT_TRUE(err.isEmpty()) << err.toStdString();
    ASSERT_EQ(frames.size(), 8);
    EXPECT_EQ(framesOf(frames), (QList<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(MovieLive, ASingleFrameCanBeExtracted)
{
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(6);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 4, 4, 1, err);
    ASSERT_TRUE(err.isEmpty()) << err.toStdString();
    ASSERT_EQ(frames.size(), 1);
    EXPECT_EQ(frameOf(frames.first()), 3) << "the single frame is not the one selected";
}

TEST_F(MovieLive, TheFramesComeBackAsAbsolutePathsInOrder)
{
    // the slide show opens them by path and shows them in the order given, so
    // both matter to the caller
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(5);
    ASSERT_FALSE(movie.isEmpty());
    const QString out = outDir();

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, out, 1, 5, 1, err);
    ASSERT_TRUE(err.isEmpty()) << err.toStdString();
    ASSERT_EQ(frames.size(), 5);
    for (const auto &f : frames) {
        EXPECT_TRUE(QDir::isAbsolutePath(f)) << f.toStdString();
        EXPECT_TRUE(QFile::exists(f)) << f.toStdString();
        EXPECT_EQ(QFileInfo(f).absolutePath(), QDir(out).absolutePath());
    }
    QStringList sorted = frames;
    sorted.sort();
    EXPECT_EQ(frames, sorted) << "the frames are not in order";
}

// ------------------------------------------------------------------ refusals

TEST_F(MovieLive, AnEmptySelectionExtractsNothingAndSaysSo)
{
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(4);
    ASSERT_FALSE(movie.isEmpty());
    const QString out = outDir();

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, out, 5, 2, 1, err);
    EXPECT_TRUE(frames.isEmpty());
    EXPECT_FALSE(err.isEmpty()) << "an impossible range was accepted silently";
    EXPECT_TRUE(QDir(out).entryList({"frame_*.png"}, QDir::Files).isEmpty());
}

TEST_F(MovieLive, AFailedExtractionLeavesNoHalfWrittenFramesBehind)
{
    // a partial sequence is worse than none: the slide show would open it and
    // present an incomplete trajectory as though it were the whole thing
    REQUIRE_FFMPEG();
    Modals modals;
    const QString bogus = dir.filePath("broken.mp4");
    QFile f(bogus);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("not a movie\n");
    f.close();
    const QString out = outDir();

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, bogus, out, 1, 4, 1, err);
    EXPECT_TRUE(frames.isEmpty());
    EXPECT_FALSE(err.isEmpty()) << "a failed extraction reported success";
    EXPECT_TRUE(QDir(out).entryList({"frame_*.png"}, QDir::Files).isEmpty())
        << "frames from a failed extraction were left in the output directory";
}

TEST_F(MovieLive, ARangeBeyondTheEndOfTheMovieStopsAtTheLastFrame)
{
    // the dialog clamps the spin boxes, but a movie whose frame count was
    // guessed from the duration can still be short by one
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(5);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 3, 20, 1, err);
    // either it extracts what exists or it reports a failure; what it must not
    // do is claim frames it does not have
    if (err.isEmpty()) {
        EXPECT_LE(frames.size(), 3) << "it produced more frames than the movie has";
        for (const auto &f : frames) EXPECT_TRUE(QFile::exists(f));
    }
}

TEST_F(MovieLive, AnExtractionThatFindsNoFramesIsReportedRatherThanReturningEmpty)
{
    // a range entirely past the end: ffmpeg is happy -- the filter simply
    // matches nothing -- and exits zero having written no files.  Returning an
    // empty list with no error would leave the caller opening an empty show
    // with nothing to explain it.
    REQUIRE_FFMPEG();
    Modals modals;
    const QString movie = makeMovie(5);
    ASSERT_FALSE(movie.isEmpty());

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, movie, outDir(), 20, 25, 1, err);
    EXPECT_TRUE(frames.isEmpty());
    EXPECT_FALSE(err.isEmpty())
        << "an extraction that produced no frames reported success";
}

TEST_F(MovieLive, StaleFramesFromAnEarlierImportAreClearedWhenThisOneFails)
{
    // the output directory can already hold frames from an import that was
    // abandoned.  If this one fails, those must go too -- otherwise the failure
    // is reported and the slide show still finds a full-looking sequence of
    // somebody else's frames sitting there.
    REQUIRE_FFMPEG();
    Modals modals;
    const QString out = outDir();
    for (int i = 1; i <= 3; ++i) {
        QImage img(8, 8, QImage::Format_RGB32);
        img.fill(Qt::magenta);
        ASSERT_TRUE(img.save(QString("%1/frame_%2.png").arg(out).arg(i, 5, 10, QChar('0'))));
    }
    ASSERT_EQ(QDir(out).entryList({"frame_*.png"}, QDir::Files).size(), 3);

    const QString bogus = dir.filePath("broken2.mp4");
    QFile f(bogus);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly));
    f.write("not a movie\n");
    f.close();

    QString err;
    const QStringList frames = extractMovieFrames(nullptr, bogus, out, 1, 3, 1, err);
    EXPECT_TRUE(frames.isEmpty());
    EXPECT_FALSE(err.isEmpty());
    EXPECT_TRUE(QDir(out).entryList({"frame_*.png"}, QDir::Files).isEmpty())
        << "the failed import left an earlier import's frames in place";
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_movielive.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
