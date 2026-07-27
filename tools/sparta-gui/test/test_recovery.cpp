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

// Crash recovery: the autosave copy of an unsaved buffer, and the offer to
// restore it on the next launch.
//
// It had no coverage at all, and it is the one feature whose failure the user
// discovers only after losing work -- there is no way to notice that autosave
// has quietly stopped writing until the session it was supposed to survive.
// The other direction matters just as much: this machinery writes files on a
// timer and restores them over the editor, so a bug that pointed it at the
// user's real deck would destroy exactly what it exists to protect.
//
// Everything here goes through the real triggers.  The write happens on the
// autosave timer (set to one second), the offer happens in the constructor, and
// the clear happens on save and on declining -- no test-only entry points.

#include "spartagui.h"

#include "codeeditor.h"
#include "constants.h"

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QIcon>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QSettings>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTimer>

#include <memory>

namespace {

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

/// Answers whatever modal appears: message boxes with @p button, file dialogs
/// with @p savePath.  Records every message so a test can say which one it got.
class Answer : public QObject {
public:
    explicit Answer(QMessageBox::StandardButton button = QMessageBox::No,
                    QString savePath = QString(), int budgetMs = 5000) :
        button(button), savePath(std::move(savePath)), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answer::poll);
        timer.start();
    }

    QMessageBox::StandardButton button;
    QString savePath;
    QStringList messages;
    int boxes = 0;
    int fileDialogs = 0;

    [[nodiscard]] bool said(const QString &needle) const
    {
        for (const auto &m : messages)
            if (m.contains(needle)) return true;
        return false;
    }
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            else if (m) m->close();
            return;
        }
        if (!m) return;
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++fileDialogs;
            if (savePath.isEmpty()) {
                static_cast<QDialog *>(fd)->reject();
            } else {
                fd->setDirectory(QFileInfo(savePath).absolutePath());
                fd->selectFile(savePath);
                static_cast<QDialog *>(fd)->accept();
            }
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            ++boxes;
            messages << box->text() + " " + box->informativeText();
            if (auto *b = box->button(button)) b->click();
            else box->reject();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
        else m->close();
    }

    QTimer timer;
    int left;
};

/// openFile/writeFile are protected on the window; the tests drive them the
/// way the File menu does.
class TestableGui : public SpartaGui {
public:
    using SpartaGui::SpartaGui;
    using SpartaGui::openFile;
    using SpartaGui::writeFile;
};

class Recovery : public ::testing::Test {
protected:
    void SetUp() override
    {
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.setValue(Keys::RESTORE_SESSION, false);
        settings.setValue(Keys::AUTOSAVE_INTERVAL, 1); // seconds
        settings.sync();

        // start every test with no leftover recovery state
        QFile::remove(recoveryPath());
        QFile::remove(recoveryPath() + ".json");

        startDir = QDir::currentPath();
        QDir::setCurrent(dir.path());
    }

    void TearDown() override
    {
        delete gui;
        gui = nullptr;
        QDir::setCurrent(startDir);
        QFile::remove(recoveryPath());
        QFile::remove(recoveryPath() + ".json");
        QSettings().clear();
    }

    /// the same path SpartaGui::recoveryFilePath() computes
    static QString recoveryPath()
    {
        return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) +
               "/recovery/session.in";
    }

    static QString read(const QString &path)
    {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
        return QString::fromUtf8(f.readAll());
    }

    static void write(const QString &path, const QString &text)
    {
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly | QIODevice::Text));
        f.write(text.toUtf8());
    }

    /// plant a recovery file as a crashed session would have left it
    static void plant(const QString &text, const QString &realPath = QString())
    {
        write(recoveryPath(), text);
        QJsonObject o;
        o["realPath"] = realPath;
        o["savedAt"]  = "2026-01-02T03:04:05";
        QFile meta(recoveryPath() + ".json");
        ASSERT_TRUE(meta.open(QIODevice::WriteOnly));
        meta.write(QJsonDocument(o).toJson());
    }

    template <class F> static bool waitFor(F done, int budgetMs)
    {
        QElapsedTimer clock;
        clock.start();
        while (!done() && clock.elapsed() < budgetMs)
            QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        return done();
    }

    /// build the window, which is where the recovery offer happens
    TestableGui *build()
    {
        gui = new TestableGui(nullptr, QString(), 800, 600);
        return gui;
    }

    CodeEditor *editor() { return gui->findChild<CodeEditor *>(); }

    void setBuffer(const QString &text)
    {
        editor()->setPlainText(text);
        editor()->document()->setModified(true); // setPlainText() clears the flag
    }

    /// wait for the autosave timer (1 s) to write the recovery file
    static bool waitForAutosave(int budgetMs = 4000)
    {
        return waitFor([] { return QFileInfo::exists(recoveryPath()); }, budgetMs);
    }

    QTemporaryDir dir;
    QString startDir;
    TestableGui *gui = nullptr;
};

} // namespace

// -------------------------------------------------------------- what it saves

TEST_F(Recovery, AnUnsavedBufferIsAutosavedWithItsText)
{
    Answer answer;
    build();
    setBuffer("run 42\n");

    ASSERT_TRUE(waitForAutosave()) << "the autosave timer never wrote a recovery file";
    EXPECT_EQ(read(recoveryPath()), "run 42\n");
}

TEST_F(Recovery, TheManifestRecordsWhereTheBufferReallyBelongs)
{
    Answer answer;
    const QString real = dir.filePath("in.real");
    write(real, "original\n");

    build();
    gui->openFile(real);
    setBuffer("edited\n");
    ASSERT_TRUE(waitForAutosave());

    const QJsonObject o = QJsonDocument::fromJson(read(recoveryPath() + ".json").toUtf8()).object();
    EXPECT_EQ(QFileInfo(o["realPath"].toString()).canonicalFilePath(),
              QFileInfo(real).canonicalFilePath())
        << "the manifest points at " << o["realPath"].toString().toStdString();
    EXPECT_FALSE(o["savedAt"].toString().isEmpty()) << "the manifest does not say when";
}

TEST_F(Recovery, AutosaveNeverTouchesTheUsersOwnFile)
{
    // the whole point of a non-destructive autosave: the file on disk stays as
    // the user last saved it, however long the editor has been left modified
    Answer answer;
    const QString real = dir.filePath("in.precious");
    write(real, "the user's own words\n");

    build();
    gui->openFile(real);
    setBuffer("scratch edits that were never saved\n");
    ASSERT_TRUE(waitForAutosave());

    EXPECT_EQ(read(real), "the user's own words\n")
        << "autosave wrote over the file it was supposed to protect";
}

TEST_F(Recovery, AnUnmodifiedBufferIsNotAutosaved)
{
    Answer answer;
    build();
    editor()->setPlainText("saved already\n"); // leaves the modified flag clear

    EXPECT_FALSE(waitForAutosave(2500))
        << "a buffer with no unsaved changes was written to the recovery file";
}

TEST_F(Recovery, AnEmptyBufferIsNotAutosaved)
{
    // recovering nothing over the user's session would be worse than not asking
    Answer answer;
    build();
    setBuffer("   \n\t\n");

    EXPECT_FALSE(waitForAutosave(2500)) << "a whitespace-only buffer was autosaved";
}

TEST_F(Recovery, SavingTheFileDropsTheRecoveryCopy)
{
    Answer answer;
    build();
    setBuffer("run 1\n");
    ASSERT_TRUE(waitForAutosave());

    gui->writeFile(dir.filePath("in.saved"));
    EXPECT_FALSE(QFileInfo::exists(recoveryPath()))
        << "the buffer matches the file on disk but a recovery copy was kept";
}

// ------------------------------------------------------------ what it restores

TEST_F(Recovery, TheOfferRestoresTheBufferAndTheFileItCameFrom)
{
    const QString real = dir.filePath("in.crashed");
    write(real, "on disk\n");
    plant("work in progress\n", real);

    Answer answer(QMessageBox::Yes);
    build();

    EXPECT_TRUE(answer.said("found autosaved work")) << answer.all().toStdString();
    EXPECT_EQ(editor()->toPlainText(), "work in progress\n") << "the buffer was not restored";
    EXPECT_TRUE(editor()->document()->isModified())
        << "recovered work was marked as saved, so closing would discard it silently";
    EXPECT_TRUE(gui->windowTitle().contains("in.crashed"))
        << "the recovered buffer forgot which file it belongs to: "
        << gui->windowTitle().toStdString();
}

TEST_F(Recovery, TheOfferNamesTheFileAndTheTimeSoTheChoiceIsInformed)
{
    plant("stuff\n", dir.filePath("in.named"));
    Answer answer(QMessageBox::No);
    build();

    EXPECT_TRUE(answer.said("in.named")) << answer.all().toStdString();
    EXPECT_TRUE(answer.said("2026-01-02")) << answer.all().toStdString();
}

TEST_F(Recovery, AnUnsavedBufferIsOfferedAsSuchRatherThanAsAFile)
{
    plant("never had a name\n"); // no realPath
    Answer answer(QMessageBox::No);
    build();
    EXPECT_TRUE(answer.said("an unsaved buffer")) << answer.all().toStdString();
}

TEST_F(Recovery, DecliningDiscardsTheCopySoItIsNotOfferedAgain)
{
    plant("unwanted\n");
    {
        Answer answer(QMessageBox::No);
        build();
        EXPECT_EQ(answer.boxes, 1);
        EXPECT_NE(editor()->toPlainText(), "unwanted\n") << "declining still restored the buffer";
    }
    EXPECT_FALSE(QFileInfo::exists(recoveryPath()))
        << "a declined recovery file survived and would be offered again";
    EXPECT_FALSE(QFileInfo::exists(recoveryPath() + ".json"))
        << "the manifest outlived the buffer it describes";
}

TEST_F(Recovery, WithNothingToRecoverNothingIsAsked)
{
    Answer answer(QMessageBox::Yes);
    build();
    EXPECT_EQ(answer.boxes, 0) << "a clean start asked about recovery anyway: "
                               << answer.all().toStdString();
}

TEST_F(Recovery, ARecoveryFileWithNoManifestIsStillRecovered)
{
    // the manifest is written second, so a crash between the two leaves the
    // buffer without one -- the text is the part worth keeping
    write(recoveryPath(), "orphaned text\n");
    Answer answer(QMessageBox::Yes);
    build();

    EXPECT_TRUE(answer.said("an unsaved buffer")) << answer.all().toStdString();
    EXPECT_EQ(editor()->toPlainText(), "orphaned text\n");
}

TEST_F(Recovery, RecoveredWorkIsAutosavedAgainWithoutBeingTouched)
{
    // it comes back modified, so the next tick has to keep protecting it
    plant("recovered\n");
    Answer answer(QMessageBox::Yes);
    build();
    ASSERT_EQ(editor()->toPlainText(), "recovered\n");

    QFile::remove(recoveryPath());
    EXPECT_TRUE(waitForAutosave()) << "recovered work stopped being autosaved";
    EXPECT_EQ(read(recoveryPath()), "recovered\n");
}

// --------------------------------------------------------------- new document

TEST_F(Recovery, StartingANewDocumentOffersToSaveTheOldOne)
{
    Answer answer(QMessageBox::No);
    build();
    setBuffer("about to be discarded\n");

    QMetaObject::invokeMethod(gui, "newDocument");
    EXPECT_GE(answer.boxes, 1) << "unsaved work was discarded without asking";
    EXPECT_FALSE(editor()->toPlainText().contains("about to be discarded"))
        << "the buffer survived a new document";
    EXPECT_FALSE(editor()->document()->isModified())
        << "a fresh document starts out modified";
}

TEST_F(Recovery, CancellingANewDocumentKeepsTheBuffer)
{
    Answer answer(QMessageBox::Cancel);
    build();
    setBuffer("keep me\n");

    QMetaObject::invokeMethod(gui, "newDocument");
    EXPECT_EQ(editor()->toPlainText(), "keep me\n")
        << "cancelling still discarded the buffer";
    EXPECT_TRUE(editor()->document()->isModified());
}

TEST_F(Recovery, ANewDocumentFromAnUnmodifiedBufferAsksNothing)
{
    Answer answer(QMessageBox::Cancel);
    build();
    editor()->setPlainText("already saved\n");

    QMetaObject::invokeMethod(gui, "newDocument");
    EXPECT_EQ(answer.boxes, 0) << "an unmodified buffer was queried anyway: "
                               << answer.all().toStdString();
    EXPECT_FALSE(editor()->toPlainText().contains("already saved"));
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    GUI_MONOFONT = std::make_unique<QFont>("Monospace", -1, QFont::Normal);
    GUI_ALLFONT  = std::make_unique<QFont>("Arial", -1, QFont::Normal);
    GUI_MONOFONT->setStyleHint(QFont::Monospace, QFont::PreferQuality);
    GUI_MONOFONT->setFixedPitch(true);
    GUI_ALLFONT->setStyleHint(QFont::SansSerif, QFont::PreferQuality);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");

    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_recovery-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());
    qputenv("XDG_DATA_HOME", settingsDir.path().toLocal8Bit());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
