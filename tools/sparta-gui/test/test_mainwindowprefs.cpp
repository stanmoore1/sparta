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

// The two main-window dialogs that write back into the running application:
// Preferences and Set Variables, plus the Open Example menu they can rebuild.
//
// Both are on test_mainwindow.cpp's do-not-trigger list because each opens a
// modal nobody was there to answer, so what happens *after* the dialog is
// accepted -- which is the whole point of both of them -- had never run.
//
// The interesting half is not that the settings are stored (the dialog does
// that itself, and test_gui_widgets covers it) but that the window acts on them
// afterwards: an examples folder that has changed rebuilds the Open Example
// menu, a preference the simulator was started with tears the instance down so
// the next run picks it up, and turning auto-lint off clears the markers it
// already put in the editor.

#include "spartagui.h"

#include "codeeditor.h"
#include "helpers.h"
#include "constants.h"
#include "inputcheck.h"
#include "preferences.h"
#include "setvariables.h"

#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QTemporaryDir>
#include <QFont>
#include <QIcon>
#include <QTimer>

#include <functional>
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

#define REQUIRE_LIBRARY()                                                                  \
    do {                                                                                   \
        if (!*testLibrary())                                                               \
            GTEST_SKIP() << "no shared libsparta: configure with -D SPARTA_TEST_LIBRARY="; \
    } while (0)

/// Runs @p fill over whichever dialog of type T appears, then accepts or
/// rejects it.  One reaper per interaction: two alive at once both answer the
/// next modal and race.
template <class T> class Answer : public QObject {
public:
    explicit Answer(std::function<void(T *)> fill, bool accept = true, int budgetMs = 8000) :
        fill(std::move(fill)), accepting(accept), left(budgetMs)
    {
        timer.setInterval(5);
        connect(&timer, &QTimer::timeout, this, &Answer::poll);
        timer.start();
    }
    int dialogs = 0;
    QStringList messages;
    [[nodiscard]] QString all() const { return messages.join(" | "); }

private:
    void poll()
    {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            timer.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            messages << box->windowTitle() + " " + box->text();
            box->accept();
            return;
        }
        if (auto *dlg = qobject_cast<T *>(m)) {
            ++dialogs;
            if (fill) fill(dlg);
            // accept()/reject() are private overrides on these dialogs; the
            // QDialog slots underneath are what a button box calls anyway
            auto *asDialog = static_cast<QDialog *>(dlg);
            if (accepting)
                asDialog->accept();
            else
                asDialog->reject();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    }
    QTimer timer;
    std::function<void(T *)> fill;
    bool accepting;
    int left;
};

class Prefs : public ::testing::Test {
protected:
    void SetUp() override
    {
        REQUIRE_LIBRARY();
        QSettings settings;
        settings.clear();
        settings.setValue(Keys::PLUGIN_PATH, QString::fromLocal8Bit(testLibrary()));
        settings.setValue(Keys::SHOWWELCOME, false);
        settings.sync();

        // Two examples the Open Example menu can find, and one it must skip
        // because it holds no input deck.
        //
        // The names are deliberately unlike anything in SPARTA's own examples
        // tree: with no configured folder the window falls back to whichever
        // tree it finds beside the shared library, and a case checking that a
        // folder was *not* picked up cannot tell "circle came from my temporary
        // directory" from "circle came from the real examples".
        ASSERT_TRUE(QDir(examples.path()).mkpath("guitest_alpha"));
        ASSERT_TRUE(QDir(examples.path()).mkpath("guitest_beta"));
        ASSERT_TRUE(QDir(examples.path()).mkpath("guitest_nodeck"));
        write(examples.filePath("guitest_alpha/in.alpha"), "dimension 2\nrun 0\n");
        write(examples.filePath("guitest_beta/in.beta"), "dimension 3\nrun 0\n");
        write(examples.filePath("guitest_nodeck/README"), "nothing to run here\n");

        bool modalSeen = false;
        QTimer reaper;
        QObject::connect(&reaper, &QTimer::timeout, [&modalSeen]() {
            if (QWidget *m = QApplication::activeModalWidget()) {
                modalSeen = true;
                m->close();
            }
        });
        reaper.start(50);
        gui = new SpartaGui(nullptr, QString(), 800, 600);
        reaper.stop();
        ASSERT_FALSE(modalSeen) << "the main window put up a modal while being constructed";
    }

    void TearDown() override
    {
        delete gui;
        gui = nullptr;
        QSettings().clear();
    }

    static void write(const QString &path, const QByteArray &bytes)
    {
        QFile f(path);
        EXPECT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(bytes);
    }

    void call(const char *slot) const
    {
        QMetaObject::invokeMethod(gui, slot);
        QCoreApplication::processEvents();
    }

    CodeEditor *editor() const { return gui->findChild<CodeEditor *>(); }

    /// How many lines carry a diagnostic marker.  diagMarks is private; the
    /// per-line backgrounds it draws are the extra selections, which are not.
    int markers() const { return editor() ? editor()->extraSelections().size() : -1; }

    /// Put an error marker on the first line and return how many markers that made.
    int markFirstLine() const
    {
        editor()->setPlainText("not_a_command 1 2 3\n");
        const int before = markers();
        InputCheck::Diagnostic d;
        d.line     = 1;
        d.severity = InputCheck::Severity::Error;
        d.message  = "unknown command";
        editor()->setDiagnostics({d});
        QCoreApplication::processEvents();
        return markers() - before;
    }

    /// The File -> Open Example submenu.
    QMenu *exampleMenu() const
    {
        for (auto *m : gui->findChildren<QMenu *>())
            if (m->title().contains("Example")) return m;
        return nullptr;
    }

    /// The names of the example decks the menu is currently offering.
    QStringList exampleEntries() const
    {
        QStringList out;
        auto *menu = exampleMenu();
        if (!menu) return out;
        for (auto *act : menu->actions())
            if (auto *sub = act->menu())
                for (auto *entry : sub->actions())
                    out << act->text() + "/" + entry->text();
        return out;
    }

    QTemporaryDir examples;
    SpartaGui *gui = nullptr;
};

} // namespace

// ------------------------------------------------------------ open example

TEST_F(Prefs, ChangingTheExamplesFolderRebuildsTheMenu)
{
    // the setting is only half the job: the menu is built once at startup, so
    // without the rebuild the user changes the folder and the old list stays
    Answer<Preferences> ok([this](Preferences *p) {
        auto *field = p->findChild<QLineEdit *>("examplesedit");
        ASSERT_NE(field, nullptr) << "no examples-folder field in the preferences";
        field->setText(examples.path());
    });
    call("preferences");
    ASSERT_EQ(ok.dialogs, 1) << "the preferences dialog never appeared: " << ok.all().toStdString();

    const QStringList entries = exampleEntries();
    EXPECT_TRUE(entries.contains("guitest_alpha/in.alpha")) << entries.join(", ").toStdString();
    EXPECT_TRUE(entries.contains("guitest_beta/in.beta")) << entries.join(", ").toStdString();
    EXPECT_EQ(entries.size(), 2) << "a directory with no input deck was offered as an example: "
                                 << entries.join(", ").toStdString();

    // and no submenu for it either: an empty one is a dead end the user opens,
    // finds nothing in, and closes again
    QStringList folders;
    for (auto *act : exampleMenu()->actions())
        if (act->menu()) folders << act->text();
    EXPECT_EQ(folders, QStringList({"guitest_alpha", "guitest_beta"}))
        << folders.join(", ").toStdString();
    EXPECT_TRUE(exampleMenu()->isEnabled());
}

TEST_F(Prefs, EveryExampleEntryCarriesThePathItOpens)
{
    // the menu entry text is the file name; what it opens comes from its data,
    // and an entry with none would silently do nothing when picked
    Answer<Preferences> ok([this](Preferences *p) {
        p->findChild<QLineEdit *>("examplesedit")->setText(examples.path());
    });
    call("preferences");
    ASSERT_EQ(ok.dialogs, 1);

    auto *menu = exampleMenu();
    ASSERT_NE(menu, nullptr);
    int checked = 0;
    for (auto *act : menu->actions()) {
        auto *sub = act->menu();
        ASSERT_NE(sub, nullptr);
        for (auto *entry : sub->actions()) {
            const QString path = entry->data().toString();
            EXPECT_FALSE(path.isEmpty()) << entry->text().toStdString() << " has no path";
            EXPECT_TRUE(QFile::exists(path)) << path.toStdString() << " does not exist";
            ++checked;
        }
    }
    EXPECT_EQ(checked, 2);
}

TEST_F(Prefs, AFolderThatIsNotAnExamplesTreeLeavesTheMenuEmpty)
{
    QTemporaryDir empty;
    Answer<Preferences> ok([&empty](Preferences *p) {
        p->findChild<QLineEdit *>("examplesedit")->setText(empty.path());
    });
    call("preferences");
    ASSERT_EQ(ok.dialogs, 1);

    // it may fall back to a tree it finds beside the library, but it must not
    // offer entries out of a directory that has none
    for (const auto &e : exampleEntries())
        EXPECT_FALSE(e.startsWith("guitest_nodeck")) << e.toStdString();
}

TEST_F(Prefs, OpeningAnExampleLoadsItIntoTheEditor)
{
    Answer<Preferences> ok([this](Preferences *p) {
        p->findChild<QLineEdit *>("examplesedit")->setText(examples.path());
    });
    call("preferences");
    ASSERT_EQ(ok.dialogs, 1);

    auto *menu = exampleMenu();
    ASSERT_NE(menu, nullptr);
    QAction *alpha = nullptr;
    for (auto *act : menu->actions())
        if (act->menu() && act->text() == "guitest_alpha")
            for (auto *entry : act->menu()->actions())
                if (entry->text() == "in.alpha") alpha = entry;
    ASSERT_NE(alpha, nullptr) << "the alpha example is not in the menu";

    alpha->trigger();
    QCoreApplication::processEvents();
    ASSERT_NE(editor(), nullptr);
    EXPECT_TRUE(editor()->toPlainText().contains("dimension 2"))
        << "the example was not loaded: " << editor()->toPlainText().toStdString();
}

// ------------------------------------------------------------- preferences

TEST_F(Prefs, CancellingThePreferencesChangesNothing)
{
    const QString before = QSettings().value(Keys::EXAMPLES_PATH, "").toString();
    Answer<Preferences> cancel(
        [this](Preferences *p) { p->findChild<QLineEdit *>("examplesedit")->setText(examples.path()); },
        /* accept= */ false);
    call("preferences");

    ASSERT_EQ(cancel.dialogs, 1);
    EXPECT_EQ(QSettings().value(Keys::EXAMPLES_PATH, "").toString(), before)
        << "a cancelled dialog stored its changes anyway";
    for (const auto &e : exampleEntries())
        EXPECT_FALSE(e.startsWith("guitest_"))
            << "a cancelled dialog rebuilt the menu anyway: " << e.toStdString();
}

TEST_F(Prefs, TurningAutoLintOffClearsTheMarkersItAlreadyLeft)
{
    // the setting alone would stop new markers appearing but leave the ones
    // already in the margin, which then never go away
    ASSERT_NE(editor(), nullptr);
    QSettings s;
    s.beginGroup(Keys::GROUP_REFORMAT);
    s.setValue(Keys::AUTOLINT, true);
    s.endGroup();
    s.sync();

    ASSERT_GT(markFirstLine(), 0) << "the marker was never placed";
    const int marked = markers();

    Answer<Preferences> off([](Preferences *p) {
        auto *box = p->findChild<QCheckBox *>("autolintval");
        ASSERT_NE(box, nullptr) << "no auto-lint control in the preferences";
        box->setChecked(false);
    });
    call("preferences");
    ASSERT_EQ(off.dialogs, 1);

    EXPECT_LT(markers(), marked) << "turning auto-lint off left its markers in the editor";
}

TEST_F(Prefs, LeavingAutoLintOnKeepsTheMarkers)
{
    ASSERT_NE(editor(), nullptr);
    QSettings s;
    s.beginGroup(Keys::GROUP_REFORMAT);
    s.setValue(Keys::AUTOLINT, true);
    s.endGroup();
    s.sync();

    ASSERT_GT(markFirstLine(), 0);
    const int marked = markers();

    Answer<Preferences> keep([](Preferences *p) {
        p->findChild<QCheckBox *>("autolintval")->setChecked(true);
    });
    call("preferences");
    ASSERT_EQ(keep.dialogs, 1);
    EXPECT_EQ(markers(), marked) << "the markers were dropped without being asked";
}

// ---------------------------------------------------------- set variables

TEST_F(Prefs, AVariableSurvivesIntoTheNextVisitToTheDialog)
{
    // `variables` is private, so what says it was kept is the dialog opening
    // with it already there -- which is also what the user sees
    {
        Answer<SetVariables> add([](SetVariables *v) {
            v->findChild<QPushButton *>("addRow")->click();
            const auto edits = v->findChildren<QLineEdit *>();
            ASSERT_GE(edits.size(), 2) << "the new row has no fields";
            edits.at(edits.size() - 2)->setText("nrho");
            edits.at(edits.size() - 1)->setText("1.0e20");
        });
        call("editVariables");
        ASSERT_EQ(add.dialogs, 1) << "the variables dialog never appeared";
    }

    QStringList seen;
    Answer<SetVariables> again([&seen](SetVariables *v) {
        for (auto *e : v->findChildren<QLineEdit *>())
            seen << e->text();
    });
    call("editVariables");
    ASSERT_EQ(again.dialogs, 1);
    EXPECT_TRUE(seen.contains("nrho")) << seen.join(", ").toStdString();
    EXPECT_TRUE(seen.contains("1.0e20")) << seen.join(", ").toStdString();
}

TEST_F(Prefs, CancellingTheVariablesDialogKeepsTheOldSet)
{
    {
        Answer<SetVariables> add([](SetVariables *v) {
            v->findChild<QPushButton *>("addRow")->click();
            const auto edits = v->findChildren<QLineEdit *>();
            edits.at(edits.size() - 2)->setText("nrho");
            edits.at(edits.size() - 1)->setText("1.0e20");
        });
        call("editVariables");
        ASSERT_EQ(add.dialogs, 1);
    }
    {
        Answer<SetVariables> discarded(
            [](SetVariables *v) {
                v->findChild<QPushButton *>("addRow")->click();
                const auto edits = v->findChildren<QLineEdit *>();
                edits.at(edits.size() - 2)->setText("fnum");
                edits.at(edits.size() - 1)->setText("7");
            },
            /* accept= */ false);
        call("editVariables");
        ASSERT_EQ(discarded.dialogs, 1);
    }

    QStringList seen;
    Answer<SetVariables> check([&seen](SetVariables *v) {
        for (auto *e : v->findChildren<QLineEdit *>())
            seen << e->text();
    });
    call("editVariables");
    ASSERT_EQ(check.dialogs, 1);
    EXPECT_TRUE(seen.contains("nrho")) << "the kept variable was lost: "
                                       << seen.join(", ").toStdString();
    EXPECT_FALSE(seen.contains("fnum")) << "a rejected variable was kept anyway: "
                                        << seen.join(", ").toStdString();
}

// -------------------------------------------------------------- help menu

// Collects the URLs the application asks the desktop to open, instead of
// letting them reach a browser that is not there.
class UrlSink : public QObject {
    Q_OBJECT
public:
    QStringList urls;
public slots:
    void take(const QUrl &u) { urls << u.toString(); }
};

TEST_F(Prefs, QuickHelpNamesTheVersionItIsDescribing)
{
    // the quick help is the one place a user reads which build they are on
    // without going near a terminal
    QStringList seen;
    QTimer poll;
    int left = 8000;
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            poll.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            seen << box->windowTitle() + "\n" + box->text() + "\n" + box->informativeText();
            box->accept();
        }
    });
    poll.setInterval(5);
    poll.start();
    call("help");
    poll.stop();

    ASSERT_FALSE(seen.isEmpty()) << "no help was shown";
    const QString text = seen.join("\n");
    EXPECT_TRUE(text.contains("Quick Help")) << text.toStdString();
    EXPECT_TRUE(text.contains(SPARTA_GUI_VERSION))
        << "the quick help does not say which version this is";
    // and it has to describe this application rather than the one it came from
    EXPECT_TRUE(text.contains("SPARTA-GUI"));
    EXPECT_FALSE(text.contains("LAMMPS"));
}

TEST_F(Prefs, TheManualAndHowToLinksPointAtTheSpartaDocumentation)
{
    // a wrong URL here is invisible in every other test: the browser is not
    // there to open it, so nothing fails
    UrlSink sink;
    QDesktopServices::setUrlHandler("https", &sink, "take");

    call("manual");
    call("howto");

    QDesktopServices::unsetUrlHandler("https");
    ASSERT_EQ(sink.urls.size(), 2) << sink.urls.join(", ").toStdString();
    EXPECT_TRUE(sink.urls.at(0).contains("Manual.html")) << sink.urls.at(0).toStdString();
    for (const auto &u : sink.urls) {
        EXPECT_TRUE(u.startsWith("https://sparta.github.io")) << u.toStdString();
        EXPECT_FALSE(u.contains("lammps")) << u.toStdString();
    }
}

// ---------------------------------------------------------- import surface

TEST_F(Prefs, ASurfaceFileThatCannotBeParsedIsRefusedRatherThanInserted)
{
    const QString junk = QDir(examples.path()).filePath("notasurface.stl");
    write(junk, "this is not an STL and not a SPARTA surface file either\n");
    ASSERT_NE(editor(), nullptr);
    editor()->setPlainText("# nothing yet\n");

    // The reaper answers the file dialog and then waits for the refusal.  It
    // must NOT reject whatever else it finds modal in the meantime: between
    // answering the file dialog and the message box appearing, the wizard --
    // built on the stack and never shown -- can be what activeModalWidget()
    // reports, and rejecting it there was enough to make the refusal never
    // arrive.  So unexpected dialogs are only recorded, and only taken down
    // once the budget is nearly gone, which is also what keeps a wizard that
    // really did open from hanging the run.
    QStringList seen;
    QStringList others;
    QTimer poll;
    constexpr int kBudget = 20000;
    int left = kBudget;
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        auto *m = QApplication::activeModalWidget();
        const bool giveUp = (left -= 5) < 500;
        if (left < 0) {
            poll.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *box = qobject_cast<QMessageBox *>(m)) {
            seen << box->windowTitle() + " " + box->text();
            box->accept();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            fd->setDirectory(QFileInfo(junk).absolutePath());
            fd->selectFile(junk);
            static_cast<QDialog *>(fd)->accept();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) {
            others << d->metaObject()->className();
            if (giveUp) d->reject();
        }
    });
    poll.setInterval(5);
    poll.start();
    call("importSurface");
    poll.stop();

    EXPECT_FALSE(seen.filter("Import Surface").isEmpty())
        << "a file it could not read was accepted without a word: "
        << seen.join(" | ").toStdString() << " (other dialogs: " << others.join(", ").toStdString()
        << ")";
    EXPECT_EQ(editor()->toPlainText(), "# nothing yet\n")
        << "something was inserted for a file that would not parse";
}

TEST_F(Prefs, CancellingTheSurfaceImportInsertsNothing)
{
    ASSERT_NE(editor(), nullptr);
    editor()->setPlainText("# nothing yet\n");

    int dialogs = 0;
    QTimer poll;
    int left = 8000;
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        auto *m = QApplication::activeModalWidget();
        if ((left -= 5) < 0) {
            poll.stop();
            if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
            return;
        }
        if (auto *fd = qobject_cast<QFileDialog *>(m)) {
            ++dialogs;
            static_cast<QDialog *>(fd)->reject();
            return;
        }
        if (auto *d = qobject_cast<QDialog *>(m)) d->reject();
    });
    poll.setInterval(5);
    poll.start();
    call("importSurface");
    poll.stop();

    EXPECT_EQ(dialogs, 1) << "it did not ask which surface to import";
    EXPECT_EQ(editor()->toPlainText(), "# nothing yet\n");
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
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
    // clears them cannot wipe the plugin path a concurrent process just wrote.
    static QTemporaryDir settingsDir;
    QCoreApplication::setOrganizationName("SPARTA-GUI test");
    QCoreApplication::setApplicationName(
        QString("test_mainwindowprefs-%1").arg(QCoreApplication::applicationPid()));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settingsDir.path());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include "test_mainwindowprefs.moc"

// Local Variables:
// c-basic-offset: 4
// End:
