// Unit tests for the pure movie-probing helpers (src/movieimport.cpp).
//
// These tests exercise parseFrameRate(), selectedFrameCount(), and
// parseProbeOutput() without running ffprobe or opening a dialog. The JSON
// samples are verbatim output of "ffprobe -of json" for an MP4 file (which
// stores the frame count) and a WebM file (which does not).

#include "movieimport.h"
#include "constants.h"
#include <QApplication>
#include <QSpinBox>
#include <QLabel>

#include <QByteArray>
#include <QString>

#include "gtest/gtest.h"

namespace {

// "ffprobe -v error -select_streams v:0 -show_entries
//  stream=width,height,nb_frames,r_frame_rate,duration -show_entries
//  format=duration -of json test.mp4" for a 4 second, 25 fps movie
const char mp4_probe[] = R"({
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "width": 320,
            "height": 240,
            "r_frame_rate": "25/1",
            "duration": "4.000000",
            "nb_frames": "100"
        }
    ],
    "format": {
        "duration": "4.000000"
    }
})";

// the same for a 2 second, 10 fps WebM movie: the container stores neither the
// frame count nor the stream duration, only the duration of the whole file
const char webm_probe[] = R"({
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "width": 320,
            "height": 240,
            "r_frame_rate": "10/1"
        }
    ],
    "format": {
        "duration": "2.000000"
    }
})";

} // namespace

TEST(ParseFrameRate, Rational)
{
    EXPECT_DOUBLE_EQ(parseFrameRate("25/1"), 25.0);
    EXPECT_DOUBLE_EQ(parseFrameRate("50/2"), 25.0);
    // NTSC rates are not integers
    EXPECT_NEAR(parseFrameRate("30000/1001"), 29.97, 1.0e-3);
}

TEST(ParseFrameRate, PlainNumber)
{
    EXPECT_DOUBLE_EQ(parseFrameRate("25"), 25.0);
    EXPECT_DOUBLE_EQ(parseFrameRate("23.976"), 23.976);
}

TEST(ParseFrameRate, Invalid)
{
    // a stream without a frame rate reports "0/0"
    EXPECT_DOUBLE_EQ(parseFrameRate("0/0"), 0.0);
    EXPECT_DOUBLE_EQ(parseFrameRate("N/A"), 0.0);
    EXPECT_DOUBLE_EQ(parseFrameRate(""), 0.0);
    EXPECT_DOUBLE_EQ(parseFrameRate("25/"), 0.0);
}

TEST(SelectedFrameCount, EveryFrame)
{
    EXPECT_EQ(selectedFrameCount(1, 1, 1), 1);
    EXPECT_EQ(selectedFrameCount(1, 100, 1), 100);
    // counting the frames from 0 must give the same result
    EXPECT_EQ(selectedFrameCount(0, 99, 1), 100);
}

TEST(SelectedFrameCount, WithInterval)
{
    // frames 1, 3, 5, 7, 9
    EXPECT_EQ(selectedFrameCount(1, 10, 2), 5);
    // frames 1, 3, 5, 7, 9, 11 -- the last one is included
    EXPECT_EQ(selectedFrameCount(1, 11, 2), 6);
    // frames 10, 15, 20, 25, 30
    EXPECT_EQ(selectedFrameCount(10, 30, 5), 5);
    // an interval larger than the range still selects the first frame
    EXPECT_EQ(selectedFrameCount(10, 12, 100), 1);
}

TEST(SelectedFrameCount, Invalid)
{
    EXPECT_EQ(selectedFrameCount(10, 9, 1), 0);
    EXPECT_EQ(selectedFrameCount(1, 10, 0), 0);
    EXPECT_EQ(selectedFrameCount(1, 10, -1), 0);
}

TEST(ParseProbeOutput, FrameCountFromContainer)
{
    const MovieInfo info = parseProbeOutput(mp4_probe);
    EXPECT_TRUE(info.valid);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_EQ(info.frames, 100);
    EXPECT_DOUBLE_EQ(info.fps, 25.0);
    EXPECT_DOUBLE_EQ(info.duration, 4.0);
    EXPECT_TRUE(info.error.isEmpty());
}

TEST(ParseProbeOutput, MissingFrameCount)
{
    const MovieInfo info = parseProbeOutput(webm_probe);
    // a movie without a stored frame count is still valid: the caller then
    // counts the video packets, which parseProbeOutput() cannot do
    EXPECT_TRUE(info.valid);
    EXPECT_EQ(info.width, 320);
    EXPECT_EQ(info.height, 240);
    EXPECT_EQ(info.frames, 0);
    EXPECT_DOUBLE_EQ(info.fps, 10.0);
    // the duration falls back to the one of the container
    EXPECT_DOUBLE_EQ(info.duration, 2.0);
    // and the frame count derived from it matches the packet count
    EXPECT_EQ(static_cast<int>(info.duration * info.fps), 20);
}

TEST(ParseProbeOutput, NoVideoStream)
{
    const MovieInfo info = parseProbeOutput(R"({"streams": [], "format": {"duration": "2.0"}})");
    EXPECT_FALSE(info.valid);
    EXPECT_FALSE(info.error.isEmpty());
}

TEST(ParseProbeOutput, NoFrameSize)
{
    const MovieInfo info = parseProbeOutput(R"({"streams": [{"r_frame_rate": "25/1"}]})");
    EXPECT_FALSE(info.valid);
    EXPECT_FALSE(info.error.isEmpty());
}

TEST(ParseProbeOutput, NotJson)
{
    const MovieInfo info = parseProbeOutput("ffprobe: command not found");
    EXPECT_FALSE(info.valid);
    EXPECT_FALSE(info.error.isEmpty());
}

TEST(ParseProbeOutput, NumbersAsJsonNumbers)
{
    // ffprobe writes nb_frames and duration as strings, but be robust in case
    // a future version writes them as JSON numbers instead
    const MovieInfo info = parseProbeOutput(
        R"({"streams": [{"width": 64, "height": 48, "r_frame_rate": "5/1",
             "nb_frames": 7, "duration": 1.4}]})");
    EXPECT_TRUE(info.valid);
    EXPECT_EQ(info.frames, 7);
    EXPECT_DOUBLE_EQ(info.duration, 1.4);
}

// ---------------------------------------------------------------- the dialog
//
// MovieImportDialog is what a user sees before a movie is unpacked into a
// folder of images.  Its estimate is the only thing standing between them and
// filling their temporary volume, so the frame arithmetic and the warning
// thresholds are worth pinning down.  The dialog needs no ffmpeg: the sample
// frame it would decode to calibrate the estimate simply comes back empty, and
// the size is reported as unknown -- which is itself a branch worth covering,
// since it is what every user without ffmpeg on their PATH gets.

namespace {

MovieInfo movie(int frames = 1000)
{
    MovieInfo m;
    m.valid    = true;
    m.width    = 640;
    m.height   = 480;
    m.frames   = frames;
    m.fps      = 25.0;
    m.duration = frames / 25.0;
    return m;
}

QSpinBox *box(MovieImportDialog &d, int n)
{
    return d.findChildren<QSpinBox *>().at(n); // first, last, interval
}

QString labelContaining(MovieImportDialog &d, const QString &needle)
{
    for (auto *l : d.findChildren<QLabel *>())
        if (l->text().contains(needle)) return l->text();
    return {};
}

} // namespace

TEST(MovieDialog, OpensOnTheWholeMovie)
{
    MovieImportDialog d("/some/where/run.mp4", movie(500));
    EXPECT_EQ(d.firstFrame(), 1);
    EXPECT_EQ(d.lastFrame(), 500) << "the dialog did not preselect every frame";
    EXPECT_EQ(d.frameInterval(), 1);
}

TEST(MovieDialog, ShowsWhatItProbed)
{
    MovieImportDialog d("/some/where/run.mp4", movie(500));
    EXPECT_FALSE(labelContaining(d, "640 x 480").isEmpty()) << "the frame size is not shown";
    EXPECT_FALSE(labelContaining(d, "25.00 frames per second").isEmpty());
    EXPECT_FALSE(labelContaining(d, "run.mp4").isEmpty()) << "the file name is not shown";
}

TEST(MovieDialog, TheRangeCannotBeInverted)
{
    MovieImportDialog d("run.mp4", movie(100));
    // dragging the first frame past the last pushes the last one along
    box(d, 0)->setValue(80);
    EXPECT_GE(d.lastFrame(), 80) << "the range was left with its end before its start";

    // and the other way round
    box(d, 1)->setValue(20);
    EXPECT_LE(d.firstFrame(), 20);
}

TEST(MovieDialog, TheFrameCountFollowsTheSelection)
{
    MovieImportDialog d("run.mp4", movie(100));
    box(d, 0)->setValue(10);
    box(d, 1)->setValue(19);
    box(d, 2)->setValue(1);
    EXPECT_EQ(selectedFrameCount(d.firstFrame(), d.lastFrame(), d.frameInterval()), 10);

    box(d, 2)->setValue(3); // every third frame of 10..19 is 10, 13, 16, 19
    EXPECT_EQ(selectedFrameCount(d.firstFrame(), d.lastFrame(), d.frameInterval()), 4);
}

TEST(MovieDialog, SaysTheSizeIsUnknownWithoutASampleFrame)
{
    // no ffmpeg decode behind it, so nothing calibrates the estimate; the
    // dialog must say so rather than show a confident zero
    MovieImportDialog d("no-such-movie.mp4", movie(100));
    bool sawUnknown = false;
    for (auto *l : d.findChildren<QLabel *>())
        if (l->text() == "unknown") sawUnknown = true;
    EXPECT_TRUE(sawUnknown) << "the size estimate was shown as a number it cannot know";
}

TEST(MovieDialog, WarnsAboutAVeryLargeExtraction)
{
    // the frame-count threshold does not need a size estimate to trip
    MovieImportDialog d("run.mp4", movie(Cfg::MOVIE_WARN_FRAMES * 4));
    EXPECT_FALSE(labelContaining(d, "Warning:").isEmpty())
        << "extracting " << Cfg::MOVIE_WARN_FRAMES * 4 << " images drew no warning";
    EXPECT_FALSE(labelContaining(d, "take a while").isEmpty());
}

TEST(MovieDialog, TheWarningClearsWhenTheSelectionShrinks)
{
    MovieImportDialog d("run.mp4", movie(Cfg::MOVIE_WARN_FRAMES * 4));
    ASSERT_FALSE(labelContaining(d, "Warning:").isEmpty());

    box(d, 1)->setValue(10); // last frame 10, so ten images
    EXPECT_TRUE(labelContaining(d, "Warning:").isEmpty())
        << "the warning stayed after the selection came back under the threshold";
    EXPECT_FALSE(labelContaining(d, "extrapolated from a single decoded frame").isEmpty())
        << "the ordinary note did not come back";
}

TEST(MovieDialog, ASingleFrameMovieIsSurvivable)
{
    MovieImportDialog d("one.mp4", movie(1));
    EXPECT_EQ(d.firstFrame(), 1);
    EXPECT_EQ(d.lastFrame(), 1);
    EXPECT_EQ(selectedFrameCount(d.firstFrame(), d.lastFrame(), d.frameInterval()), 1);
}

int main(int argc, char **argv)
{
    // the dialog cases need a QApplication; the parser cases do not, but one
    // binary cannot have it both ways
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
