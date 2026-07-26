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

// The Output panel.
//
// Everything SPARTA prints during a run lands here, and the panel's whole job
// beyond showing it is to notice the lines that matter: it keeps a running
// count of warnings and errors in a corner badge and can jump the cursor to the
// next one. On a run that produced thousands of lines that badge is the only
// thing telling the user whether anything went wrong, and nothing tested it.
//
// The counting itself belongs to FlagWarnings, which is tested separately. What
// is tested here is the panel's use of it: that text arriving in the ways a run
// actually delivers it gets counted, that the badge says so, and that stepping
// through the warnings reaches each one and wraps around rather than stopping
// at the last.

#include <gtest/gtest.h>

#include <QApplication>
#include <QLabel>
#include <QRegularExpression>
#include <QSettings>
#include <QTextCursor>

#include "logwindow.h"

namespace {

// The badge, which is the only place a count is shown.
QString badge(LogWindow &w)
{
    for (auto *l : w.findChildren<QLabel *>())
        if (l->text().contains("Lines")) return l->text();
    return QString();
}

int countIn(const QString &text, const QString &what)
{
    static const QRegularExpression re("(\\d+)\\s+" + QRegularExpression::escape(what));
    const auto m = re.match(text);
    return m.hasMatch() ? m.captured(1).toInt() : -1;
}

int warningsShown(LogWindow &w)
{
    return countIn(badge(w), "Warnings");
}

int linesShown(LogWindow &w)
{
    const QRegularExpression re("-\\s+(\\d+)\\s+Lines");
    const auto m = re.match(badge(w));
    return m.hasMatch() ? m.captured(1).toInt() : -1;
}

// A run delivers its output in chunks, appended at the end, exactly as
// SpartaGui::logUpdate() does it.
void feed(LogWindow &w, const QString &chunk)
{
    w.moveCursor(QTextCursor::End);
    w.insertPlainText(chunk);
    w.moveCursor(QTextCursor::End);
    QCoreApplication::processEvents();
}

} // namespace

class Log : public ::testing::Test {
protected:
    void SetUp() override
    {
        QCoreApplication::setOrganizationName("SPARTA-GUI test");
        QCoreApplication::setApplicationName("test_logwindow");
        QSettings().clear();
    }
    void TearDown() override { QSettings().clear(); }
};

TEST_F(Log, ComesUpEmptyAndSaysSo)
{
    LogWindow w("in.circle", nullptr);
    EXPECT_TRUE(w.toPlainText().isEmpty());
    EXPECT_EQ(badge(w), QString("0 Warnings / Errors - 0 Lines"))
        << "the badge should start at zero rather than blank";
}

TEST_F(Log, ClearOutputIsCountedAsNoProblems)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "Step CPU Np\n0 0 0\n100 0.5 43387\n");

    EXPECT_EQ(warningsShown(w), 0) << "ordinary stats output was counted as a problem";
    EXPECT_GT(linesShown(w), 0) << "the line count did not follow the text";
}

TEST_F(Log, AWarningIsCountedAndShown)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "Step CPU\n0 0\nWARNING: Using compute with no output\n100 0.5\n");

    EXPECT_EQ(warningsShown(w), 1)
        << "a WARNING line in the output was not counted, so the badge says a run with "
           "problems went cleanly";
}

TEST_F(Log, AnErrorIsCountedToo)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "ERROR: Unknown command: fom\n");
    EXPECT_EQ(warningsShown(w), 1) << "an ERROR line was not counted";
}

TEST_F(Log, TheCountFollowsTheOutputAsItArrives)
{
    LogWindow w("in.circle", nullptr);

    feed(w, "Step CPU\n");
    EXPECT_EQ(warningsShown(w), 0);

    feed(w, "WARNING: one\n");
    EXPECT_EQ(warningsShown(w), 1);

    feed(w, "0 0\n1 1\n");
    EXPECT_EQ(warningsShown(w), 1) << "clean lines after a warning discarded the count";

    feed(w, "ERROR: two\n");
    EXPECT_EQ(warningsShown(w), 2)
        << "a second problem did not raise the count, so a run with many is indistinguishable "
           "from a run with one";
}

// "warning" inside a sentence is not a warning line; SPARTA's own output says
// things like "no warnings" and the badge must not count those.
TEST_F(Log, OnlyLinesThatStartWithWarningOrErrorAreCounted)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "this line mentions a WARNING in passing\n"
            "  WARNING: indented, still a warning\n"
            "no errors were found\n");

    EXPECT_LE(warningsShown(w), 1)
        << "a line merely mentioning the word was counted as a problem of its own";
}

// The badge button and Ctrl+N step through the problems. On a long log this is
// the only way to reach them.
TEST_F(Log, SteppingThroughTheWarningsReachesEachOne)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "0 0\nWARNING: first\n1 1\n2 2\nWARNING: second\n3 3\n");
    ASSERT_EQ(warningsShown(w), 2);

    w.moveCursor(QTextCursor::Start);

    QList<int> visited;
    for (int i = 0; i < 2; ++i) {
        QMetaObject::invokeMethod(&w, "nextWarning");
        visited << w.textCursor().blockNumber();
    }

    ASSERT_EQ(visited.size(), 2);
    EXPECT_NE(visited[0], visited[1])
        << "stepping twice landed on the same warning, so the second is unreachable";
}

TEST_F(Log, SteppingPastTheLastWarningWrapsAround)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "0 0\nWARNING: only one\n1 1\n2 2\n");
    ASSERT_EQ(warningsShown(w), 1);

    w.moveCursor(QTextCursor::Start);
    QMetaObject::invokeMethod(&w, "nextWarning");
    const int first = w.textCursor().blockNumber();

    // from beyond it, the next step has to come back round rather than stop
    w.moveCursor(QTextCursor::End);
    QMetaObject::invokeMethod(&w, "nextWarning");
    EXPECT_EQ(w.textCursor().blockNumber(), first)
        << "the cursor did not wrap around, so the button stops working once the last "
           "warning is behind it";
}

TEST_F(Log, SteppingWithNoWarningsLeavesTheCursorAlone)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "0 0\n1 1\n2 2\n");
    w.moveCursor(QTextCursor::Start);
    const int before = w.textCursor().blockNumber();

    QMetaObject::invokeMethod(&w, "nextWarning");
    EXPECT_EQ(w.textCursor().blockNumber(), before)
        << "with nothing to jump to the cursor moved anyway";
}

// The panel is created read-only by the application, and typing into a
// transcript of what SPARTA printed would make it a record of nothing.
TEST_F(Log, CanBeMadeReadOnlyAndStaysThatWayAsOutputArrives)
{
    LogWindow w("in.circle", nullptr);
    w.setReadOnly(true);
    feed(w, "WARNING: something\n0 0\n");

    EXPECT_TRUE(w.isReadOnly()) << "appending output re-enabled editing";
    EXPECT_EQ(warningsShown(w), 1) << "a read-only panel stopped counting";
}

TEST_F(Log, ARunWithNoNameIsStillUsable)
{
    LogWindow w(QString(), nullptr);
    feed(w, "WARNING: no deck name\n");
    EXPECT_EQ(warningsShown(w), 1);
}

TEST_F(Log, SurvivesAVeryLongLineAndAVeryLargeLog)
{
    LogWindow w("in.circle", nullptr);
    feed(w, QString("x").repeated(20000) + "\n");
    QString bulk;
    for (int i = 0; i < 2000; ++i)
        bulk += QString("%1 %2 %3\n").arg(i).arg(i * 0.5).arg(i * 2);
    feed(w, bulk);
    feed(w, "WARNING: after all that\n");

    EXPECT_EQ(warningsShown(w), 1) << "the count was lost somewhere in a large log";
    EXPECT_GT(linesShown(w), 2000);
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
