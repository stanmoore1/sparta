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

// The welcome screen.
//
// This is the first thing a new user sees, and it had no test. It is driven
// entirely by two setters and answers with four signals, so what matters is
// that it filters what it is given -- a recent-files list is a list of paths
// that may since have been deleted or moved -- and that picking something
// reports the path the user actually picked rather than the display name.
//
// The screen is deliberately excluded from the widget walker, because it is a
// full-window overlay that covers everything the walker goes on to drive.

#include <gtest/gtest.h>

#include <QAbstractButton>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QIcon>
#include <QLabel>
#include <QListWidget>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTextStream>

#include <memory>

#include "welcomescreen.h"

namespace {

void touch(const QString &path, const QString &text = "# deck\n")
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream(&f) << text;
    }
}

// The two lists, told apart by which one holds recents. rebuildRecents() hides
// the recents list when it is empty, so identify by object rather than by
// visibility.
QList<QListWidget *> lists(WelcomeScreen &w)
{
    return w.findChildren<QListWidget *>();
}

} // namespace

class Welcome : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_TRUE(dir.isValid()); }
    QTemporaryDir dir;
};

TEST_F(Welcome, HasTwoListsAndComesUpEmpty)
{
    WelcomeScreen w;
    ASSERT_EQ(lists(w).size(), 2) << "the welcome screen should offer recents and examples";
    for (auto *l : lists(w))
        EXPECT_EQ(l->count(), 0) << "a list arrived populated before being given anything";
}

TEST_F(Welcome, ARecentFileThatExistsIsOffered)
{
    const QString deck = dir.filePath("in.circle");
    touch(deck);

    WelcomeScreen w;
    w.setRecentFiles({deck});

    int found = 0;
    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i)
            if (l->item(i)->data(Qt::UserRole).toString() == deck) ++found;
    EXPECT_EQ(found, 1) << "a recent file that is really there was not offered";
}

// The recent list is whatever was open last time, and files move and get
// deleted between sessions. Offering one that is gone gives an error dialog
// instead of a document.
TEST_F(Welcome, ARecentFileThatIsGoneIsNotOffered)
{
    const QString here = dir.filePath("in.here");
    touch(here);
    const QString gone = dir.filePath("in.gone");

    WelcomeScreen w;
    w.setRecentFiles({here, gone});

    QStringList offered;
    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i)
            offered << l->item(i)->data(Qt::UserRole).toString();

    EXPECT_TRUE(offered.contains(here));
    EXPECT_FALSE(offered.contains(gone))
        << "a recent file that no longer exists is still on the welcome screen";
}

TEST_F(Welcome, TheRecentsAreLabelledByNameAndCarryTheWholePath)
{
    const QString deck = dir.filePath("in.circle");
    touch(deck);

    WelcomeScreen w;
    w.setRecentFiles({deck});

    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i) {
            auto *item = l->item(i);
            if (item->data(Qt::UserRole).toString() != deck) continue;
            EXPECT_EQ(item->text(), QString("in.circle"))
                << "the entry is labelled with a path rather than a file name";
            EXPECT_EQ(item->toolTip(), deck)
                << "the tooltip should say which in.circle this is";
        }
}

TEST_F(Welcome, SettingTheRecentsAgainReplacesRatherThanAppends)
{
    const QString a = dir.filePath("in.a");
    const QString b = dir.filePath("in.b");
    touch(a);
    touch(b);

    WelcomeScreen w;
    w.setRecentFiles({a});
    w.setRecentFiles({b});

    QStringList offered;
    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i)
            offered << l->item(i)->data(Qt::UserRole).toString();

    EXPECT_TRUE(offered.contains(b));
    EXPECT_FALSE(offered.contains(a))
        << "the previous recent-files list was appended to rather than replaced";
}

// The empty state has to say something: an invisible list and no label is a
// blank column with no explanation.
TEST_F(Welcome, TheEmptyRecentsStateExplainsItself)
{
    WelcomeScreen w;
    w.setRecentFiles({});

    bool explained = false;
    for (auto *l : w.findChildren<QLabel *>())
        if (!l->text().isEmpty() && l->isVisibleTo(&w)) explained = true;
    EXPECT_TRUE(explained) << "with no recent files the column says nothing at all";
}

TEST_F(Welcome, AnEmptyOrMissingExamplesDirectoryIsHarmless)
{
    WelcomeScreen w;
    w.setExamplesDir(QString());
    w.setExamplesDir("/nonexistent/examples");
    w.setExamplesDir(dir.path());   // real, but holds no example subdirectories
    SUCCEED() << "surviving this is the assertion";
}

// The gallery enumerates the examples tree the same way the File menu does:
// each in.* in each subdirectory, skipping bench. Only decks with a shipped
// thumbnail are shown, so a tree of decks with no thumbnails yields nothing --
// which is the state a user's own directory would be in.
TEST_F(Welcome, TheGalleryOnlyOffersDecksItCanIllustrate)
{
    touch(dir.filePath("circle/in.circle"));
    touch(dir.filePath("bench/in.bench"));
    touch(dir.filePath("mine/in.mine"));

    WelcomeScreen w;
    w.setExamplesDir(dir.path());

    QStringList offered;
    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i)
            offered << l->item(i)->data(Qt::UserRole).toString();

    for (const QString &path : offered) {
        EXPECT_FALSE(path.contains("/bench/"))
            << "the benchmark directory is not a gallery of examples";
        EXPECT_TRUE(QFile::exists(path)) << "the gallery offers " << path.toStdString()
                                         << ", which is not there";
    }
}

TEST_F(Welcome, SettingTheSameExamplesDirectoryTwiceIsNotADoubleGallery)
{
    touch(dir.filePath("circle/in.circle"));

    WelcomeScreen w;
    w.setExamplesDir(dir.path());
    int first = 0;
    for (auto *l : lists(w))
        first += l->count();

    w.setExamplesDir(dir.path());
    int second = 0;
    for (auto *l : lists(w))
        second += l->count();

    EXPECT_EQ(first, second) << "the gallery was rebuilt on top of itself";
}

// Picking an entry has to report the path, not the label: two examples in
// different directories are both called in.circle.
//
// A single click, not a double click or Enter: this is a launcher screen and
// its lists are wired to itemClicked.
TEST_F(Welcome, ClickingARecentReportsItsWholePath)
{
    const QString deck = dir.filePath("in.circle");
    touch(deck);

    WelcomeScreen w;
    w.setRecentFiles({deck});
    QSignalSpy spy(&w, &WelcomeScreen::openFileRequested);

    for (auto *l : lists(w))
        for (int i = 0; i < l->count(); ++i)
            if (l->item(i)->data(Qt::UserRole).toString() == deck) emit l->itemClicked(l->item(i));

    ASSERT_EQ(spy.count(), 1) << "clicking a recent file asked for nothing to be opened";
    EXPECT_EQ(spy.at(0).at(0).toString(), deck)
        << "the screen reported the label rather than the path";
}

TEST_F(Welcome, TheNewAndBrowseButtonsAskForWhatTheySay)
{
    WelcomeScreen w;
    QSignalSpy newFile(&w, &WelcomeScreen::newFileRequested);
    QSignalSpy browse(&w, &WelcomeScreen::browseRequested);

    for (auto *b : w.findChildren<QAbstractButton *>()) {
        const QString t = b->text();
        if (t.contains("New", Qt::CaseInsensitive)) b->click();
        else if (t.contains("Open", Qt::CaseInsensitive) || t.contains("Browse", Qt::CaseInsensitive))
            b->click();
    }

    EXPECT_EQ(newFile.count(), 1) << "nothing on the welcome screen starts a new file";
    EXPECT_EQ(browse.count(), 1) << "nothing on the welcome screen browses for a file";
}

int main(int argc, char **argv)
{
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    Q_INIT_RESOURCE(spartagui);
    QIcon::setThemeSearchPaths(QStringList() << ":/icons");
    QIcon::setThemeName("spartagui");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
