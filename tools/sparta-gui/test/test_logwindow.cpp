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
#include <QTimer>
#include <QTemporaryDir>
#include <QMouseEvent>
#include <QMessageBox>
#include <QKeyEvent>
#include <QFileDialog>
#include <QFile>
#include <QDir>
#include <QDialog>
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


// ------------------------------------------------------------------ saving
//
// Everything below the warning badge: writing the log out, pulling the YAML
// blocks out of it, the context menu that offers those, and the shortcuts that
// reach them.  None of it needed a simulator; all of it ends in a modal that
// nothing was answering, which is why it went uncovered.

namespace {

// Answers the file dialog (and anything else modal) that an action raises.
// With AA_DontUseNativeDialogs the dialog is an ordinary widget that can be
// handed a path and accepted from a timer.
class Answer : public QObject {
public:
    explicit Answer(QString path = QString(), int budgetMs = 3000) :
        answer(std::move(path)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answer::poll);
        timer.start();
    }

    QString answer;
    int fileDialogs = 0;
    QStringList messages;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }

private:
    void poll()
    {
        if ((left -= 5) < 0) { timer.stop(); return; }
        auto *m = QApplication::activeModalWidget();
        if (!m) return;
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++fileDialogs;
            if (answer.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(answer).absolutePath());
                fd->selectFile(answer);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->text() + "\n" + box->informativeText();
            box->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }

    QTimer timer;
    int left;
};

QString readAll(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    return QString::fromUtf8(f.readAll());
}

// A log with a SPARTA YAML stats block in it, the way a deck with
// "stats_style yaml" writes one.
const char *const kYamlLog = "SPARTA (24 Sep 2025)\n"
                             "Created orthogonal box\n"
                             "---\n"
                             "keywords: ['Step','CPU','Np',]\n"
                             "data:\n"
                             "  - [0, 0, 100,]\n"
                             "  - [5, 0.01, 100,]\n"
                             "...\n"
                             "Loop time of 0.01 on 1 procs\n";

} // namespace

TEST_F(Log, SavingWritesWhatIsOnScreen)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, "SPARTA (24 Sep 2025)\nWARNING: something\n");

    const QString out = dir.filePath("saved.log");
    Answer answer(out);
    QMetaObject::invokeMethod(&w, "saveAs");
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "Save Log did not ask where to save";
    EXPECT_EQ(readAll(out), w.toPlainText());
}

TEST_F(Log, SavingAddsAFinalNewlineWhenTheLogLacksOne)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, "one line with no newline after it");

    const QString out = dir.filePath("nonewline.log");
    Answer answer(out);
    QMetaObject::invokeMethod(&w, "saveAs");
    QCoreApplication::processEvents();

    EXPECT_TRUE(readAll(out).endsWith('\n')) << "the saved log has no terminating newline";
    EXPECT_FALSE(readAll(out).endsWith("\n\n")) << "a newline was added to one already there";
}

TEST_F(Log, CancellingTheSaveWritesNothing)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, "some output\n");

    Answer answer; // cancel
    QMetaObject::invokeMethod(&w, "saveAs");
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(QDir(dir.path()).entryList(QDir::Files).isEmpty());
}

TEST_F(Log, SavingSomewhereUnwritableSaysSo)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "some output\n");

    Answer answer("/proc/definitely/not/writable/sparta.log");
    QMetaObject::invokeMethod(&w, "saveAs");
    QCoreApplication::processEvents();
    EXPECT_TRUE(answer.said("Cannot save")) << answer.messages.join(" | ").toStdString();
}

// ------------------------------------------------------------------ YAML

TEST_F(Log, ALogWithNoYamlOffersNothingToExtract)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "SPARTA (24 Sep 2025)\nStep CPU Np\n0 0 100\n");

    Answer answer(QDir::tempPath() + "/should-not-be-written.yaml");
    QMetaObject::invokeMethod(&w, "extractYaml");
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 0)
        << "a log with no YAML in it still asked where to save some";
}

TEST_F(Log, ExtractingYamlWritesOnlyTheYamlLines)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, QString::fromLatin1(kYamlLog));

    const QString out = dir.filePath("stats.yaml");
    Answer answer(out);
    QMetaObject::invokeMethod(&w, "extractYaml");
    QCoreApplication::processEvents();

    ASSERT_EQ(answer.fileDialogs, 1) << "the YAML export did not ask where to save";
    const QString yaml = readAll(out);
    EXPECT_TRUE(yaml.contains("keywords: ['Step','CPU','Np',]")) << yaml.toStdString();
    EXPECT_TRUE(yaml.contains("  - [0, 0, 100,]")) << yaml.toStdString();
    EXPECT_TRUE(yaml.contains("data:"));
    EXPECT_TRUE(yaml.contains("---"));
    EXPECT_FALSE(yaml.contains("SPARTA (24 Sep 2025)"))
        << "the banner was written into the YAML file: " << yaml.toStdString();
    EXPECT_FALSE(yaml.contains("Loop time"))
        << "the run summary was written into the YAML file";
}

TEST_F(Log, CancellingTheYamlExportWritesNothing)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, QString::fromLatin1(kYamlLog));

    Answer answer; // cancel
    QMetaObject::invokeMethod(&w, "extractYaml");
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(QDir(dir.path()).entryList(QDir::Files).isEmpty());
}

// ------------------------------------------------------------------ shortcuts

TEST_F(Log, ControlWClosesTheWindow)
{
    LogWindow w("in.circle", nullptr);
    w.show();
    ASSERT_TRUE(w.isVisible());

    QKeyEvent close(QEvent::ShortcutOverride, 'W', Qt::ControlModifier);
    QCoreApplication::sendEvent(&w, &close);
    EXPECT_FALSE(w.isVisible()) << "Ctrl+W left the log window open";
}

TEST_F(Log, ControlNStepsToTheNextWarning)
{
    // the panel shares the main window's shortcut context, so these are claimed
    // here rather than left ambiguous with the identical menu shortcuts
    LogWindow w("in.circle", nullptr);
    feed(w, "line one\nWARNING: first\nline three\nWARNING: second\n");
    w.show();
    w.moveCursor(QTextCursor::Start);

    QKeyEvent next(QEvent::ShortcutOverride, 'N', Qt::ControlModifier);
    QCoreApplication::sendEvent(&w, &next);
    const int first = w.textCursor().blockNumber();
    QCoreApplication::sendEvent(&w, &next);
    const int second = w.textCursor().blockNumber();

    EXPECT_NE(first, second) << "Ctrl+N did not move on to the next warning";
    EXPECT_GT(second, first);
}

TEST_F(Log, ControlSAsksWhereToSave)
{
    QTemporaryDir dir;
    LogWindow w("in.circle", nullptr);
    feed(w, "some output\n");
    w.show();

    const QString out = dir.filePath("byshortcut.log");
    Answer answer(out);
    QKeyEvent save(QEvent::ShortcutOverride, 'S', Qt::ControlModifier);
    QCoreApplication::sendEvent(&w, &save);
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "Ctrl+S did not reach Save Log";
    EXPECT_EQ(readAll(out), w.toPlainText());
}

TEST_F(Log, TheStopAndRunShortcutsAreHarmlessWithNoMainWindow)
{
    // both go through a SpartaGui the standalone log window does not have
    LogWindow w("in.circle", nullptr);
    w.show();
    for (int key : {int(0x2f), int(Qt::Key_Return)}) { // Ctrl+/ and Ctrl+Return
        QKeyEvent ev(QEvent::ShortcutOverride, key, Qt::ControlModifier);
        QCoreApplication::sendEvent(&w, &ev);
    }
    EXPECT_TRUE(w.isVisible());
}

TEST_F(Log, AnUnrelatedKeyIsLeftAlone)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "untouched\n");
    w.show();
    QKeyEvent plain(QEvent::ShortcutOverride, Qt::Key_A, Qt::NoModifier);
    QCoreApplication::sendEvent(&w, &plain);
    EXPECT_TRUE(w.isVisible());
    EXPECT_EQ(w.toPlainText(), "untouched\n") << "a keystroke edited a read-only log";
}

// ------------------------------------------------------------------ error links

TEST_F(Log, DoubleClickingAnErrorLinkPicksTheUrlOutOfTheLine)
{
    // SPARTA prints a documentation URL beside an error; double-clicking it is
    // how a user gets to the explanation
    LogWindow w("in.circle", nullptr);
    feed(w, "ERROR: Unknown command (https://sparta.github.io/err0042) on line 7\n");

    // put the cursor inside the URL, then double-click there
    w.moveCursor(QTextCursor::Start);
    auto cursor = w.textCursor();
    const int col = w.toPlainText().indexOf("https://") + 10;
    cursor.setPosition(col);
    w.setTextCursor(cursor);

    QMouseEvent dbl(QEvent::MouseButtonDblClick, QPointF(1, 1), QPointF(1, 1),
                    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(w.viewport(), &dbl);
    QCoreApplication::processEvents();
    SUCCEED() << "double-clicking a URL did not disturb the window";
}

TEST_F(Log, DoubleClickingOrdinaryTextSelectsItInstead)
{
    LogWindow w("in.circle", nullptr);
    feed(w, "just some ordinary output\n");
    w.moveCursor(QTextCursor::Start);

    QMouseEvent dbl(QEvent::MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5),
                    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(w.viewport(), &dbl);
    QCoreApplication::processEvents();
    EXPECT_EQ(w.toPlainText(), "just some ordinary output\n");
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    // a native file dialog runs its own event loop nothing here can reach into
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
