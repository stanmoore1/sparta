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

// The menu walker. The command palette, the generated shortcut sheet and the
// status-tip pass all read the menu bar through scanMenuBar(), so what it
// returns decides what those features can see. Its failure modes are quiet
// ones -- a skipped submenu level just makes some actions unfindable -- so
// each rule it claims to follow is pinned here.

#include "actionscan.h"

#include <gtest/gtest.h>

#include <QAction>
#include <QApplication>
#include <QMenu>
#include <QMenuBar>

namespace {

// a menu bar with every construct the walker must handle: nesting, separators,
// mnemonics, a disabled action, an empty-text action
QMenuBar *makeBar()
{
    auto *bar  = new QMenuBar;
    auto *file = bar->addMenu("&File");
    file->addAction("&Open");
    file->addSeparator();
    auto *sub = file->addMenu("Open &Example");
    auto *deep = sub->addMenu("circle");
    deep->addAction("in.circle");
    auto *dis = file->addAction("&Disabled");
    dis->setEnabled(false);
    file->addAction(""); // unassigned recent-file slot
    auto *edit = bar->addMenu("&Edit");
    edit->addAction("Cut && &Paste");
    return bar;
}

} // namespace

TEST(ActionScan, StripsMnemonicsAndKeepsLiteralAmpersands)
{
    EXPECT_EQ(strippedActionText("&Save").toStdString(), "Save");
    EXPECT_EQ(strippedActionText("Cut && &Paste").toStdString(), "Cut & Paste");
    EXPECT_EQ(strippedActionText("no mnemonic").toStdString(), "no mnemonic");
}

TEST(ActionScan, WalksEveryLevelAndRecordsThePath)
{
    QMenuBar *bar = makeBar();
    const auto infos = scanMenuBar(bar);

    QStringList found;
    for (const auto &i : infos) found << (i.path + " / " + i.text);

    EXPECT_TRUE(found.contains("File / Open"));
    // two submenu levels deep: the level a walker without recursion loses
    EXPECT_TRUE(found.contains("File > Open Example > circle / in.circle"))
        << found.join("; ").toStdString();
    EXPECT_TRUE(found.contains("Edit / Cut & Paste"));
    delete bar;
}

TEST(ActionScan, KeepsDisabledActionsAndSkipsTheEmptyAndTheStructural)
{
    QMenuBar *bar = makeBar();
    const auto infos = scanMenuBar(bar);

    bool sawDisabled = false, sawEmpty = false, sawSubmenuHeader = false;
    for (const auto &i : infos) {
        if (i.text == "Disabled") sawDisabled = true;
        if (i.text.isEmpty()) sawEmpty = true;
        // a submenu's own action opens the submenu; it is not a command
        if (i.text == "Open Example") sawSubmenuHeader = true;
    }
    EXPECT_TRUE(sawDisabled) << "disabled actions are still part of what the app can do";
    EXPECT_FALSE(sawEmpty) << "an unassigned recent-file slot is not an action";
    EXPECT_FALSE(sawSubmenuHeader);
    delete bar;
}

int main(int argc, char **argv)
{
    // offscreen: gtest_discover_tests also executes this binary at build time,
    // where there is no display
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv); // QMenu needs a QApplication
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Local Variables:
// c-basic-offset: 4
// End:
