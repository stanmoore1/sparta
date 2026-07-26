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

// The read-only file viewer, the plain window shell the two viewers share, and
// the band slider.  Three small classes that the first coverage measurement
// found at nothing at all: 66, 25 and 48 lines, none of them executed by any
// test in any configuration.
//
// The file viewer is the one that matters.  It is what "View output file" and
// the log/dump browsers open, it decompresses on the fly through six external
// programs, and the branch where the file cannot be read at all is what a user
// sees when a run wrote nothing.

#include "fileviewer.h"
#include "rangebandslider.h"
#include "viewerwindow.h"

#include <gtest/gtest.h>

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QKeyEvent>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTextStream>

namespace {

const char *const kText = "Step Temp Press\n0 300.0 1.0e5\n100 301.5 1.1e5\n";

// A scratch directory per test, so a leftover file cannot make the next one
// pass for the wrong reason.
class Files : public ::testing::Test {
protected:
    QString write(const QString &name, const QByteArray &bytes) const
    {
        const QString path = dir.filePath(name);
        QFile f(path);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
        return path;
    }

    // Compress kText with @p program, or return an empty string when that
    // program is not installed -- the test then says it skipped rather than
    // failing on the machine's package list.
    QString compressed(const QString &name, const QString &program,
                       const QStringList &args = {}) const
    {
        if (QStandardPaths::findExecutable(program).isEmpty()) return {};
        const QString path = dir.filePath(name);
        QProcess p;
        p.setStandardOutputFile(path);
        p.start(program, args.isEmpty() ? QStringList{"-c"} : args);
        if (!p.waitForStarted()) return {};
        p.write(kText);
        p.closeWriteChannel();
        if (!p.waitForFinished(10000) || p.exitCode() != 0) return {};
        return path;
    }

    QTemporaryDir dir;
};

} // namespace

// ---------------------------------------------------------------- plain files

TEST_F(Files, ShowsThePlainTextItWasGiven)
{
    FileViewer v(write("thermo.txt", kText), nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
    EXPECT_TRUE(v.isReadOnly()) << "the viewer would let a user edit a log file";
    EXPECT_EQ(v.lineWrapMode(), QPlainTextEdit::NoWrap)
        << "wrapped columns make a thermo table unreadable";
}

TEST_F(Files, StartsAtTheTopRatherThanTheEnd)
{
    QByteArray big;
    for (int i = 0; i < 500; ++i)
        big += QByteArray::number(i) + " line\n";
    FileViewer v(write("big.txt", big), nullptr);
    EXPECT_EQ(v.textCursor().position(), 0);
}

TEST_F(Files, TitlesItselfAfterTheFileUnlessToldOtherwise)
{
    const QString path = write("thermo.txt", kText);
    FileViewer plain(path, nullptr);
    EXPECT_TRUE(plain.windowTitle().contains(path)) << plain.windowTitle().toStdString();

    FileViewer titled(path, nullptr, "Log of run 3");
    EXPECT_EQ(titled.windowTitle(), "Log of run 3");
}

TEST_F(Files, AnEmptyFileIsAnEmptyWindowNotAnError)
{
    FileViewer v(write("empty.txt", QByteArray()), nullptr);
    EXPECT_TRUE(v.toPlainText().isEmpty());
}

// ---------------------------------------------------------------- the failure path

TEST_F(Files, SaysWhyItCouldNotOpenTheFile)
{
    // what a user sees when a run wrote no output: the reason, in the window,
    // rather than a blank one
    FileViewer v(dir.filePath("was-never-written.txt"), nullptr);
    const QString shown = v.toPlainText();
    EXPECT_TRUE(shown.contains("Could not open")) << shown.toStdString();
    EXPECT_TRUE(shown.contains("was-never-written.txt")) << "the message omits which file";
    EXPECT_GT(shown.size(), QString("Could not open file : ").size())
        << "the message omits the reason";
}

TEST_F(Files, AMisnamedCompressedFileIsShownAsPlainTextAnyway)
{
    // The decompressors are invoked with -f, which makes them copy input they
    // cannot decompress straight through.  So a .gz that is not gzip data is
    // shown as the text it actually is rather than as an error -- which is the
    // forgiving behaviour, and the reason the flag is there.  Dropping the -f
    // would turn every misnamed file into an empty window.
    FileViewer v(write("misnamed.gz", kText), nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

// ---------------------------------------------------------------- decompression

TEST_F(Files, ReadsGzip)
{
    const QString path = compressed("thermo.txt.gz", "gzip");
    if (path.isEmpty()) GTEST_SKIP() << "gzip is not installed";
    FileViewer v(path, nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

TEST_F(Files, ReadsBzip2)
{
    const QString path = compressed("thermo.txt.bz2", "bzip2");
    if (path.isEmpty()) GTEST_SKIP() << "bzip2 is not installed";
    FileViewer v(path, nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

TEST_F(Files, ReadsXz)
{
    const QString path = compressed("thermo.txt.xz", "xz");
    if (path.isEmpty()) GTEST_SKIP() << "xz is not installed";
    FileViewer v(path, nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

TEST_F(Files, ReadsLzma)
{
    // the one entry in the table that needs an extra argument, so the one that
    // breaks if the argument is inserted in the wrong position
    const QString path = compressed("thermo.txt.lzma", "xz", {"-c", "--format=lzma"});
    if (path.isEmpty()) GTEST_SKIP() << "xz is not installed";
    FileViewer v(path, nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

TEST_F(Files, ReadsZstd)
{
    const QString path = compressed("thermo.txt.zst", "zstd");
    if (path.isEmpty()) GTEST_SKIP() << "zstd is not installed";
    FileViewer v(path, nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

TEST_F(Files, AnUnknownSuffixIsReadAsPlainText)
{
    // .log is not in the compression table, so it must not be piped anywhere
    FileViewer v(write("run.log", kText), nullptr);
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText));
}

// ---------------------------------------------------------------- shortcuts

TEST_F(Files, ControlWClosesTheWindow)
{
    FileViewer v(write("thermo.txt", kText), nullptr);
    v.show();
    ASSERT_TRUE(v.isVisible());

    QKeyEvent close(QEvent::ShortcutOverride, 'W', Qt::ControlModifier);
    QCoreApplication::sendEvent(&v, &close);
    EXPECT_FALSE(v.isVisible()) << "Ctrl+W left the viewer open";
}

TEST_F(Files, TheStopShortcutIsHarmlessWithNoMainWindow)
{
    // stopRun() goes through a SpartaGui the standalone viewer does not have
    FileViewer v(write("thermo.txt", kText), nullptr);
    v.show();
    QKeyEvent stop(QEvent::ShortcutOverride, '/', Qt::ControlModifier);
    QCoreApplication::sendEvent(&v, &stop);
    EXPECT_TRUE(v.isVisible()) << "Ctrl+/ closed the viewer";
}

TEST_F(Files, OtherKeysLeaveItOpen)
{
    FileViewer v(write("thermo.txt", kText), nullptr);
    v.show();
    QKeyEvent plain(QEvent::ShortcutOverride, Qt::Key_A, Qt::NoModifier);
    QCoreApplication::sendEvent(&v, &plain);
    EXPECT_TRUE(v.isVisible());
    EXPECT_EQ(v.toPlainText(), QString::fromLatin1(kText)) << "a keystroke edited a read-only view";
}

// ---------------------------------------------------------------- the band slider

TEST(BandSlider, PaintsWhateverRangeItIsGiven)
{
    RangeBandSlider s;
    s.setRange(0, 100);
    s.resize(200, 30);

    // the ordinary case, the inverted case, and the degenerate one: the widget
    // paints a two-colour track from these, so a division by the band width is
    // where it would fail
    for (const auto &band : {QPair<int, int>{20, 80}, {80, 20}, {50, 50}, {0, 0}, {-10, 200}}) {
        s.setActiveRange(band.first, band.second);
        EXPECT_FALSE(s.grab().isNull()) << band.first << ".." << band.second;
    }
}

TEST(BandSlider, PaintsWithAZeroWidthScale)
{
    RangeBandSlider s;
    s.setRange(5, 5); // min == max: the scale has no extent to map onto
    s.resize(200, 30);
    s.setActiveRange(5, 5);
    EXPECT_FALSE(s.grab().isNull());
}

// ---------------------------------------------------------------- the window shell

TEST(Shell, WrapsAViewerAndFollowsItsTitle)
{
    // ViewerWindow is a shell: a central widget, an icon, a minimum size, and
    // two connections. Build it around a slide show with nothing to show --
    // the case where a user opens the viewer before any image exists.
    QTemporaryDir dir;
    auto *win = ViewerWindow::forSequence(dir.filePath("no-such-image.0.ppm"), nullptr);
    ASSERT_NE(win, nullptr);
    EXPECT_NE(win->sequence(), nullptr) << "a sequence window did not hold a slide show";
    EXPECT_EQ(win->snapshot(), nullptr) << "a sequence window claimed to hold an image viewer";
    EXPECT_TRUE(win->windowTitle().contains("Slide Show")) << win->windowTitle().toStdString();
    EXPECT_NE(win->centralWidget(), nullptr);
    delete win;
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("sparta-gui-test");
    QCoreApplication::setApplicationName(
        QStringLiteral("test_fileviewer.%1").arg(QCoreApplication::applicationPid()));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
