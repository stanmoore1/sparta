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

// The main window's File menu, and the workers behind it.
//
// test_mainwindow.cpp keeps a list of actions it must not trigger, because each
// opens a modal dialog that would sit there with nobody to answer it -- and
// nearly all of them are the file actions.  That is why opening, saving,
// viewing and inspecting a file were the largest uncovered block in the
// application.
//
// Two things make them reachable.  The workers -- openFile(), writeFile(),
// viewFile(), inspectFile() -- are public and take the path directly, so what
// each menu entry actually *does* needs no dialog at all.  And the thin wrapper
// that asks for the path can be driven by answering the QFileDialog from a
// timer, which is what the driver below does: with AA_DontUseNativeDialogs the
// dialog is an ordinary Qt widget, so it can be handed a filename and accepted.

#include "spartagui.h"

#include "codeeditor.h"
#include "constants.h"
#include "fileviewer.h"

#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QTemporaryDir>
#include <QTextStream>
#include <QFont>
#include <QIcon>
#include <QTimer>

#include <memory>

namespace {

const char *const kDeck = "# a small deck\n"
                          "dimension       2\n"
                          "global          gridcut 0.0\n"
                          "run             0\n";

// Where to find a shared libsparta. Baked in by the build; the environment
// still wins so the binary can be run by hand against another one.
const char *testLibrary()
{
    static const QByteArray env = qgetenv("SPARTA_PLUGIN_LIB");
    if (!env.isEmpty()) return env.constData();
#if defined(SPARTA_TEST_LIBRARY_PATH)
    return SPARTA_TEST_LIBRARY_PATH;
#else
    return "";
#endif
}

#define REQUIRE_LIBRARY()                                                                  \
    do {                                                                                   \
        if (!*testLibrary())                                                               \
            GTEST_SKIP() << "no shared libsparta: configure with -D SPARTA_TEST_LIBRARY="; \
    } while (0)

// Answers the file dialog (and anything else modal) that an action raises.
//
// A modal spins its own event loop, so nothing in the test body runs until it
// is dismissed; a timer polling for the active modal is the only way in. With
// AA_DontUseNativeDialogs a QFileDialog is an ordinary widget, so it can be
// handed a path and accepted -- which is what makes the two-line wrappers over
// QFileDialog::getOpenFileName reachable at all.
class FileAnswer : public QObject {
public:
    /// @param answer the path to hand the dialog; empty cancels it
    explicit FileAnswer(QString answer = QString(), int budgetMs = 3000) :
        path(std::move(answer)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &FileAnswer::poll);
        timer.start();
    }

    QString path;
    int fileDialogs = 0; ///< how many file dialogs were answered
    int others      = 0; ///< how many other modals were dismissed
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
        if ((left -= 5) < 0) {
            timer.stop();
            return;
        }
        auto *m = QApplication::activeModalWidget();
        if (!m) return;

        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++fileDialogs;
            if (path.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(path).absolutePath());
                fd->selectFile(path);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            ++others;
            messages << box->text() + "\n" + box->informativeText();
            box->accept();
            return;
        }
        ++others;
        if (auto *d = qobject_cast<QDialog *>(m))
            d->reject();
        else
            m->close();
    }

    QTimer timer;
    int left;
};

// The workers are protected, which is right -- nothing outside the window has
// business calling them.  A subclass is how a test reaches a protected member
// without widening the interface for everyone else.
class TestableGui : public SpartaGui {
public:
    using SpartaGui::SpartaGui;
    using SpartaGui::inspectFile;
    using SpartaGui::openFile;
    using SpartaGui::viewFile;
    using SpartaGui::writeFile;
};

class Files : public ::testing::Test {
protected:
    void SetUp() override
    {
        REQUIRE_LIBRARY();
        QSettings settings;
        settings.clear();
        // without this the constructor loops on its "No SPARTA Shared Library"
        // box, which offscreen is a hang rather than a failure
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.sync();

        startDir = QDir::currentPath();

        // The window's constructor loops on its "No SPARTA Shared Library" box
        // until someone answers, and offscreen nobody can -- so a settings
        // problem would show up as a test that never returns. Turn it into a
        // failure instead.
        bool modalSeen = false;
        QTimer reaper;
        QObject::connect(&reaper, &QTimer::timeout, [&modalSeen]() {
            if (QWidget *m = QApplication::activeModalWidget()) {
                modalSeen = true;
                m->close();
            }
        });
        reaper.start(50);
        gui = new TestableGui(nullptr, QString(), 800, 600);
        reaper.stop();
        ASSERT_FALSE(modalSeen) << "the main window put up a modal while being constructed";
    }

    void TearDown() override
    {
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir); // writeFile() chdirs; do not leak that
        QSettings().clear();
    }

    QString write(const QString &name, const QByteArray &bytes) const
    {
        const QString p = dir.filePath(name);
        QFile f(p);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
        f.close();
        return p;
    }

    CodeEditor *editor() const { return gui->findChild<CodeEditor *>(); }

    QString bufferText() const { return editor() ? editor()->toPlainText() : QString(); }

    /// Put @p text in the editor as an unsaved edit.  setPlainText() resets the
    /// document's modified flag, so a test that wants a dirty buffer -- which
    /// is what every save path branches on -- has to say so.
    void setBuffer(const QString &text) const
    {
        if (!editor()) return;
        editor()->setPlainText(text);
        editor()->document()->setModified(true);
    }

    /// Every menu action, walked rather than found: an action in a submenu is
    /// not a child of the menu bar, so findChildren() misses most of them.
    static void collect(QMenu *menu, QList<QAction *> &out)
    {
        for (auto *a : menu->actions()) {
            out.append(a);
            if (a->menu()) collect(a->menu(), out);
        }
    }

    QList<QAction *> allActions() const
    {
        QList<QAction *> out;
        if (auto *bar = gui->findChild<QMenuBar *>())
            for (auto *top : bar->actions()) {
                out.append(top);
                if (top->menu()) collect(top->menu(), out);
            }
        return out;
    }

    /// trigger the menu action whose text is @p name, wherever it lives
    bool trigger(const QString &name) const
    {
        for (auto *act : allActions())
            if (act->text() == name) {
                act->trigger();
                return true;
            }
        return false;
    }

    /// the standalone viewers an action opened, so a test can close them
    static QList<QWidget *> extraTopLevels(const QList<QWidget *> &before)
    {
        QList<QWidget *> out;
        for (auto *w : QApplication::topLevelWidgets())
            if (!before.contains(w)) out << w;
        return out;
    }

    QTemporaryDir dir;
    QString startDir;
    TestableGui *gui = nullptr;
};

} // namespace

// ---------------------------------------------------------------- opening

TEST_F(Files, OpeningADeckPutsItInTheEditor)
{
    const QString deck = write("in.small", kDeck);
    gui->openFile(deck);
    EXPECT_EQ(bufferText(), QString::fromLatin1(kDeck));
    EXPECT_TRUE(gui->windowTitle().contains("in.small")) << gui->windowTitle().toStdString();
}

TEST_F(Files, OpeningADeckMakesItsFolderTheWorkingDirectory)
{
    // every relative path in the deck is resolved against this, so it is the
    // difference between a run that finds its data files and one that does not
    const QString deck = write("in.small", kDeck);
    gui->openFile(deck);
    EXPECT_EQ(QDir::currentPath(), QFileInfo(deck).absolutePath());
}

TEST_F(Files, OpeningRemembersTheFileForTheRecentList)
{
    const QString deck = write("in.recent", kDeck);
    gui->openFile(deck);

    bool listed = false;
    for (auto *act : allActions())
        if (act->data().toString().contains("in.recent")) listed = true;
    EXPECT_TRUE(listed) << "the file did not reach the Open Recent list";
}

TEST_F(Files, OpeningNothingIsANoOp)
{
    setBuffer("untouched");
    gui->openFile(QString()); // what a cancelled dialog hands back
    EXPECT_EQ(bufferText(), "untouched");
}

TEST_F(Files, OpeningAFileThatIsNotThereSaysSo)
{
    FileAnswer answer;
    gui->openFile(dir.filePath("in.absent"));
    QCoreApplication::processEvents();
    EXPECT_TRUE(answer.said("annot")) << answer.messages.join(" | ").toStdString();
}

TEST_F(Files, TheOpenActionAsksForAFileAndOpensIt)
{
    // the two-line wrapper the menu is wired to, driven through its dialog
    const QString deck = write("in.viadialog", kDeck);
    FileAnswer answer(deck);
    ASSERT_TRUE(trigger("&Open Input File"));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "the Open action did not ask for a file";
    EXPECT_EQ(bufferText(), QString::fromLatin1(kDeck));
}

TEST_F(Files, CancellingTheOpenDialogChangesNothing)
{
    setBuffer("untouched");
    FileAnswer answer; // empty path: cancel
    ASSERT_TRUE(trigger("&Open Input File"));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_EQ(bufferText(), "untouched") << "cancelling Open cleared the editor";
}

// ---------------------------------------------------------------- saving

TEST_F(Files, WritingTheBufferProducesTheFile)
{
    setBuffer("run 100\n");
    const QString out = dir.filePath("in.written");
    gui->writeFile(out);

    QFile f(out);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    EXPECT_EQ(QString::fromUtf8(f.readAll()), "run 100\n");
}

TEST_F(Files, WritingAddsTheFinalNewlineIfTheBufferLacksOne)
{
    // SPARTA's parser wants a terminated last line; an editor buffer often has
    // no trailing newline because nobody pressed Return at the end
    setBuffer("run 100");
    const QString out = dir.filePath("in.nonewline");
    gui->writeFile(out);

    QFile f(out);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    EXPECT_EQ(QString::fromUtf8(f.readAll()), "run 100\n");
}

TEST_F(Files, WritingDoesNotDoubleAnExistingFinalNewline)
{
    setBuffer("run 100\n");
    const QString out = dir.filePath("in.onenewline");
    gui->writeFile(out);

    QFile f(out);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    EXPECT_EQ(QString::fromUtf8(f.readAll()), "run 100\n");
}

TEST_F(Files, WritingClearsTheModifiedFlagAndRetitlesTheWindow)
{
    setBuffer("run 100\n");
    ASSERT_TRUE(editor()->document()->isModified());

    gui->writeFile(dir.filePath("in.saved"));
    EXPECT_FALSE(editor()->document()->isModified())
        << "the buffer still looks unsaved after being saved";
    EXPECT_TRUE(gui->windowTitle().contains("in.saved")) << gui->windowTitle().toStdString();
}

TEST_F(Files, WritingNowhereIsANoOp)
{
    setBuffer("run 100\n");
    gui->writeFile(QString()); // what a cancelled Save As hands back
    EXPECT_TRUE(editor()->document()->isModified())
        << "a cancelled save marked the buffer as saved";
}

TEST_F(Files, WritingSomewhereUnwritableSaysSoAndChangesNothing)
{
    setBuffer("run 100\n");
    const QString before = gui->windowTitle();

    FileAnswer answer;
    gui->writeFile("/proc/definitely/not/writable/in.nope");
    QCoreApplication::processEvents();

    EXPECT_TRUE(answer.said("Cannot save")) << answer.messages.join(" | ").toStdString();
    EXPECT_EQ(gui->windowTitle(), before)
        << "the window took the name of a file it failed to write";
    EXPECT_TRUE(editor()->document()->isModified())
        << "a failed save marked the buffer as saved";
}

TEST_F(Files, SaveAsAsksForAPathAndWritesThere)
{
    setBuffer("run 42\n");
    const QString out = dir.filePath("in.saveas");
    FileAnswer answer(out);
    ASSERT_TRUE(trigger("Save Input File &As"));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "Save As did not ask where to save";
    QFile f(out);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    EXPECT_EQ(QString::fromUtf8(f.readAll()), "run 42\n");
}

TEST_F(Files, CancellingSaveAsWritesNothing)
{
    setBuffer("run 42\n");
    FileAnswer answer;
    ASSERT_TRUE(trigger("Save Input File &As"));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(editor()->document()->isModified()) << "a cancelled Save As claimed to have saved";
}

TEST_F(Files, SaveReusesTheNameTheFileAlreadyHasWithoutAsking)
{
    // the distinction between Save and Save As: once a deck has a name, Save
    // must not put a dialog in the way
    const QString deck = write("in.named", kDeck);
    gui->openFile(deck);
    setBuffer("run 7\n");

    FileAnswer answer;
    ASSERT_TRUE(trigger("&Save Input File"));
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 0) << "Save asked for a name the deck already had";

    QFile f(deck);
    ASSERT_TRUE(f.open(QIODevice::ReadOnly | QIODevice::Text));
    EXPECT_EQ(QString::fromUtf8(f.readAll()), "run 7\n");
}

TEST_F(Files, ARoundTripThroughDiskIsLossless)
{
    const QString text = "# comment with  spacing\nvariable t equal 300.0\nrun 100\n";
    setBuffer(text);
    const QString out = dir.filePath("in.roundtrip");
    gui->writeFile(out); // this clears the modified flag, so no prompt below
    editor()->setPlainText("something else entirely");

    gui->openFile(out);
    EXPECT_EQ(bufferText(), text);
}

TEST_F(Files, OpeningOverUnsavedEditsAsksFirst)
{
    // the guard that stands between a user and losing an afternoon's work
    const QString other = write("in.other", kDeck);
    setBuffer("edits nobody has saved\n");
    ASSERT_TRUE(editor()->document()->isModified());

    FileAnswer answer; // dismisses the prompt
    gui->openFile(other);
    QCoreApplication::processEvents();
    EXPECT_GE(answer.others, 1)
        << "a file was opened over unsaved edits without asking";
}

// ---------------------------------------------------------------- viewing

TEST_F(Files, ViewingAFileOpensAReadOnlyWindowOnIt)
{
    const auto before  = QApplication::topLevelWidgets();
    const QString note = write("output.txt", "Step Temp\n0 300\n");
    gui->viewFile(note);
    QCoreApplication::processEvents();

    const auto opened = extraTopLevels(before);
    FileViewer *viewer = nullptr;
    for (auto *w : opened)
        if (auto *v = qobject_cast<FileViewer *>(w)) viewer = v;
    ASSERT_NE(viewer, nullptr) << "no viewer window appeared";
    EXPECT_EQ(viewer->toPlainText(), "Step Temp\n0 300\n");
    EXPECT_TRUE(viewer->isReadOnly());
    for (auto *w : opened)
        w->close();
}

TEST_F(Files, ViewingNothingOpensNothing)
{
    const auto before = QApplication::topLevelWidgets();
    gui->viewFile(QString());
    QCoreApplication::processEvents();
    EXPECT_TRUE(extraTopLevels(before).isEmpty());
}

TEST_F(Files, TheViewActionAsksForAFileAndShowsIt)
{
    const auto before  = QApplication::topLevelWidgets();
    const QString note = write("log.txt", "hello\n");
    FileAnswer answer(note);
    ASSERT_TRUE(trigger("&View Text File"));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "the View action did not ask for a file";
    const auto opened = extraTopLevels(before);
    EXPECT_FALSE(opened.isEmpty()) << "nothing was shown";
    for (auto *w : opened)
        w->close();
}

TEST_F(Files, CancellingTheViewDialogOpensNothing)
{
    const auto before = QApplication::topLevelWidgets();
    FileAnswer answer;
    ASSERT_TRUE(trigger("&View Text File"));
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(extraTopLevels(before).isEmpty()) << "a cancelled View opened a window anyway";
}

// ---------------------------------------------------------------- images

TEST_F(Files, TheImageActionOpensASlideShowOnWhatItWasGiven)
{
    QImage img(32, 24, QImage::Format_RGB32);
    img.fill(Qt::darkCyan);
    const QString png = dir.filePath("snap.0001.png");
    ASSERT_TRUE(img.save(png));

    const auto before = QApplication::topLevelWidgets();
    FileAnswer answer(png);
    ASSERT_TRUE(trigger("View &Image or Movie File(s)..."));
    QCoreApplication::processEvents();

    EXPECT_EQ(answer.fileDialogs, 1) << "the image action did not ask for a file";
    const auto opened = extraTopLevels(before);
    EXPECT_FALSE(opened.isEmpty()) << "no viewer window appeared";
    for (auto *w : opened)
        w->close();
}

TEST_F(Files, CancellingTheImageDialogOpensNothing)
{
    const auto before = QApplication::topLevelWidgets();
    FileAnswer answer;
    ASSERT_TRUE(trigger("View &Image or Movie File(s)..."));
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1);
    EXPECT_TRUE(extraTopLevels(before).isEmpty());
}

// ---------------------------------------------------------------- restart files

TEST_F(Files, InspectingNothingIsANoOp)
{
    const auto before = QApplication::topLevelWidgets();
    gui->inspectFile(QString());
    QCoreApplication::processEvents();
    EXPECT_TRUE(extraTopLevels(before).isEmpty());
}

TEST_F(Files, InspectingSomethingThatIsNotARestartFileSaysSo)
{
    // a text file with a restart-ish name: read_restart must refuse it rather
    // than open an empty inspection window
    const QString fake = write("tmp.restart", "this is not a SPARTA restart file\n");
    FileAnswer answer;
    gui->inspectFile(fake);
    QCoreApplication::processEvents();

    EXPECT_GE(answer.others, 1) << "a bad restart file was accepted in silence: "
                                << answer.messages.join(" | ").toStdString();
}

TEST_F(Files, TheInspectActionAsksForAFile)
{
    FileAnswer answer; // cancel it: reading a real restart needs a run first
    ASSERT_TRUE(trigger("Inspect &Restart File"));
    QCoreApplication::processEvents();
    EXPECT_EQ(answer.fileDialogs, 1) << "the Inspect action did not ask for a file";
}

// ---------------------------------------------------------------- other modals

TEST_F(Files, TheSnippetPickerOffersSomethingAndInsertsIt)
{
    setBuffer("");
    // the picker is a modal list; take whatever it defaults to
    class Taker : public QObject {
    public:
        Taker()
        {
            timer.setInterval(5);
            connect(&timer, &QTimer::timeout, this, [this]() {
                if ((left -= 5) < 0) { timer.stop(); return; }
                if (auto *m = QApplication::activeModalWidget()) {
                    ++seen;
                    if (auto *d = qobject_cast<QDialog *>(m)) d->accept();
                    else m->close();
                }
            });
            timer.start();
        }
        int seen = 0;

    private:
        QTimer timer;
        int left = 3000;
    } taker;

    ASSERT_TRUE(trigger("Insert &Snippet..."));
    QCoreApplication::processEvents();
    EXPECT_GE(taker.seen, 1) << "the snippet picker never appeared";
}

TEST_F(Files, TheAboutBoxComesUpAndGoesAway)
{
    FileAnswer answer;
    ASSERT_TRUE(trigger("&About SPARTA-GUI"));
    QCoreApplication::processEvents();
    EXPECT_GE(answer.others, 1) << "the About box never appeared";
}

TEST_F(Files, TheWindowSurvivesEveryFileActionInSequence)
{
    // the actions test_mainwindow's BLOCKING list skips, run back to back
    const QString deck = write("in.sequence", kDeck);
    for (const char *name : {"&Open Input File", "&View Text File", "Inspect &Restart File",
                             "View &Image or Movie File(s)...", "Save Input File &As"}) {
        FileAnswer answer; // cancel each
        ASSERT_TRUE(trigger(QLatin1String(name))) << name;
        QCoreApplication::processEvents();
        EXPECT_EQ(answer.fileDialogs, 1) << name << " did not ask for a file";
    }
    EXPECT_NE(editor(), nullptr) << "the window lost its editor along the way";
    EXPECT_TRUE(gui->isEnabled());
}

int main(int argc, char **argv)
{
    // offscreen, so this needs no display and no window manager
    qputenv("QT_QPA_PLATFORM", "offscreen");
    // the whole point: a native file dialog runs its own event loop that
    // nothing here can reach into, while Qt's own is an ordinary widget
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    // main() is deliberately not linked here, so this stands in for the part of
    // it the window reads while constructing itself.
    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT  = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    // Settings of this process's own, in a directory of its own, so a case that
    // clears them cannot wipe the plugin path a concurrent process just wrote
    // and leave it sitting on the missing-library dialog.
    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_mainwindowfiles-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
